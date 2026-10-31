import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from scfuse.model.encoders import _tf
from scfuse.utils.evm import N_OPCODES

HEADS = ("concat", "attention", "prompt", "shaspec", "vulnsense", "cmml", "qmf", "pdf")


def _bias(base_rate: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(torch.logit(base_rate.float().clamp(0.01, 0.99)))


class ConcatHead(nn.Module):
    def __init__(self, K: int, d: int, n_views: int, base_rate: torch.Tensor):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.randn(K, n_views * d) * 0.02), _bias(base_rate)

    def forward(self, c, m):
        return {"logit": torch.einsum("bkf,kf->bk", c.permute(0, 2, 1, 3).flatten(2).float(), self.w.float()) + self.b}


class AttentionHead(nn.Module):
    def __init__(self, K: int, d: int, base_rate: torch.Tensor):
        super().__init__()
        self.q = nn.Parameter(torch.randn(K, d) * 0.02)
        self.w, self.b = nn.Parameter(torch.randn(K, d) * 0.02), _bias(base_rate)

    def forward(self, c, m):
        score = torch.einsum("bvkd,kd->bvk", c.float(), self.q.float()).masked_fill(m[:, :, None] == 0, -1e4)
        fused = (score.softmax(1).unsqueeze(-1) * c.float()).sum(1)
        return {"logit": torch.einsum("bkd,kd->bk", fused, self.w.float()) + self.b}


class PromptHead(nn.Module):
    def __init__(self, K: int, d: int, n_views: int, prompt_len: int, layers: int, heads: int, dropout: float,
                 base_rate: torch.Tensor):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(2 ** n_views, prompt_len, d) * 0.02)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d))
        self.vtype = nn.Parameter(torch.randn(n_views, d) * 0.02)
        self.tf, self.norm = _tf(d, layers, heads, dropout), nn.LayerNorm(d)
        self.w, self.b = nn.Parameter(torch.randn(K, d) * 0.02), _bias(base_rate)
        self.register_buffer("bits", 2 ** torch.arange(n_views), persistent=False)

    def forward(self, c, m):
        B, V, K, d = c.shape
        avail = m > 0
        tok = (c + self.vtype[None, :, None].to(c.dtype)).permute(0, 2, 1, 3).reshape(B * K, V, d)
        prompts = self.prompts[(avail.long() * self.bits).sum(1)].repeat_interleave(K, 0).to(tok.dtype)
        x = torch.cat([self.cls_tok.expand(B * K, 1, d).to(tok.dtype), prompts, tok], 1)
        pad = torch.cat([avail.new_zeros(B * K, 1 + prompts.shape[1]), ~avail.repeat_interleave(K, 0)], 1)
        h = self.norm(self.tf(x, src_key_padding_mask=pad))[:, 0].view(B, K, d)
        return {"logit": (h.float() * self.w.float()).sum(-1) + self.b}


class ShaSpecHead(nn.Module):
    def __init__(self, K: int, d: int, n_views: int, align: float, domain: float, base_rate: torch.Tensor):
        super().__init__()
        self.shared = nn.Linear(d, d)
        self.spec = nn.ModuleList(nn.Linear(d, d) for _ in range(n_views))
        self.compose, self.dom = nn.Linear(2 * d, d), nn.Linear(d, n_views)
        self.w, self.b = nn.Parameter(torch.randn(K, n_views * d) * 0.02), _bias(base_rate)
        self.align, self.domain = align, domain

    def forward(self, c, m):
        B, V, K, d = c.shape
        avail = m > 0
        sh = self.shared(c)
        sp = torch.stack([self.spec[v](c[:, v]) for v in range(V)], 1)
        fused = sh + self.compose(torch.cat([sh, sp], -1))
        first = sh[torch.arange(B, device=c.device), avail.float().argmax(1)]
        fused = torch.where(avail[:, :, None, None], fused, first.unsqueeze(1))
        logit = torch.einsum("bkf,kf->bk", fused.permute(0, 2, 1, 3).flatten(2).float(), self.w.float()) + self.b
        aux = c.new_zeros((), dtype=torch.float32)
        for v in range(V):
            w = (v + 1) % V
            both = avail[:, v] & avail[:, w]
            if both.any():
                aux = aux + self.align * (sh[both, v] - sh[both, w]).abs().float().mean()
        bi, vi = avail.nonzero(as_tuple=True)
        if len(bi):
            aux = aux + self.domain * F.cross_entropy(self.dom(sp[bi, vi]).float().flatten(0, 1),
                                                      vi.repeat_interleave(K))
        return {"logit": logit, "aux": aux}


def opcode_sequence(x, max_len: int):
    flat = x.flatten(1)
    keep = flat != 0
    pos = keep.long().cumsum(1) - 1
    keep &= pos < max_len
    L = int(min(max_len, max(1, int(keep.sum(1).max()))))
    seq = flat.new_zeros(len(flat), L)
    seq[keep.nonzero(as_tuple=True)[0], pos[keep]] = flat[keep]
    return seq, keep.sum(1).clamp_min(1)


class VulnSenseHead(nn.Module):
    def __init__(self, K: int, d: int, hidden: int, max_ops: int, base_rate: torch.Tensor, emb: int = 64):
        super().__init__()
        self.emb = nn.Embedding(N_OPCODES + 2, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.max_ops = max_ops
        self.w, self.b = nn.Parameter(torch.randn(K, 2 * d + 2 * hidden) * 0.02), _bias(base_rate)

    def forward(self, c, m, x):
        seq, n = opcode_sequence(x, self.max_ops)
        with torch.autocast(device_type=c.device.type, enabled=False):
            packed = pack_padded_sequence(self.emb(seq).float(), n.cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
        ops = torch.cat([h[0], h[1]], -1)
        z = torch.cat([c[:, 0] * m[:, 0, None, None], c[:, 1] * m[:, 1, None, None]], -1).float()
        feat = torch.cat([z, ops.unsqueeze(1).expand(-1, z.shape[1], -1)], -1)
        return {"logit": torch.einsum("bkf,kf->bk", feat, self.w.float()) + self.b}


def _bernoulli_kl(p, q, eps: float = 1e-6):
    p, q = p.clamp(eps, 1 - eps), q.clamp(eps, 1 - eps)
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


class CMMLHead(nn.Module):
    def __init__(self, K: int, d: int, rec: float, mutual: float, base_rate: torch.Tensor):
        super().__init__()
        self.infer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.w_s, self.b_s = nn.Parameter(torch.randn(K, 2 * d) * 0.02), _bias(base_rate)
        self.w_t, self.b_t = nn.Parameter(torch.randn(K, 2 * d) * 0.02), _bias(base_rate)
        self.rec, self.mutual = rec, mutual

    def forward(self, c, m):
        zS, zG = (c[:, 0] * m[:, 0, None, None]).float(), c[:, 1].float()
        zS_hat = self.infer(zG)
        student = torch.einsum("bkf,kf->bk", torch.cat([zS_hat, zG], -1), self.w_s.float()) + self.b_s
        teacher = torch.einsum("bkf,kf->bk", torch.cat([zS, zG], -1), self.w_t.float()) + self.b_t
        has_s = m[:, 0] > 0
        aux = c.new_zeros((), dtype=torch.float32)
        if has_s.any():
            aux = aux + self.rec * (zS_hat[has_s] - zS[has_s].detach()).pow(2).mean()
        ps, pt = torch.sigmoid(student), torch.sigmoid(teacher)
        aux = aux + self.mutual * (_bernoulli_kl(pt.detach(), ps) + _bernoulli_kl(ps.detach(), pt)).mean()
        return {"logit": student, "teacher_logit": teacher, "aux": aux}


class _ViewClassifiers(nn.Module):
    def __init__(self, K: int, d: int, n_views: int, base_rate: torch.Tensor):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_views, K, d) * 0.02)
        self.b = nn.Parameter(torch.logit(base_rate.float().clamp(0.01, 0.99)).repeat(n_views, 1))

    def forward(self, c):
        return torch.einsum("bvkd,vkd->bvk", c.float(), self.w.float()) + self.b


class QMFHead(nn.Module):
    def __init__(self, K: int, d: int, n_views: int, base_rate: torch.Tensor):
        super().__init__()
        self.views = _ViewClassifiers(K, d, n_views, base_rate)

    def forward(self, c, m):
        z = self.views(c)
        avail = m > 0
        conf = torch.logaddexp(z / 2, -z / 2)
        return {"logit": (conf.detach() * z * avail[..., None]).sum(1), "view_logit": z, "view_mask": avail,
                "qmf_conf": (conf * avail[..., None]).mean(-1)}


class PDFHead(nn.Module):
    def __init__(self, K: int, d: int, n_views: int, base_rate: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.views = _ViewClassifiers(K, d, n_views, base_rate)
        self.conf = nn.ModuleList(nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))
                                  for _ in range(n_views))

    def forward(self, c, m):
        z = self.views(c)
        avail = m > 0
        a = avail[..., None].float()
        p_hat = torch.stack([torch.sigmoid(self.conf[v](c[:, v].float())).squeeze(-1) for v in range(c.shape[1])], 1)
        loss_hat = -p_hat.clamp_min(1e-6).log() * a
        holo = (loss_hat.sum(1, keepdim=True) - loss_hat) / loss_hat.sum(1, keepdim=True).clamp_min(1e-6)
        du = (torch.sigmoid(z) - 0.5).abs() * a
        n = a.sum(1, keepdim=True)
        others = (du.sum(1, keepdim=True) - du) / (n - 1).clamp_min(1)
        calib = torch.where((du < others) & (n > 1), du / others.clamp_min(1e-6), torch.ones_like(du))
        w = torch.softmax(((p_hat + holo) * calib).masked_fill(~avail[..., None], -1e4), 1).detach()
        return {"logit": (w * z * a).sum(1), "view_logit": z, "view_mask": avail, "pdf_conf": p_hat}


def build_head(name: str, K: int, d: int, n_views: int, bc: dict, dropout: float, base_rate: torch.Tensor):
    if name == "concat":
        return ConcatHead(K, d, n_views, base_rate)
    if name == "attention":
        return AttentionHead(K, d, base_rate)
    if name == "prompt":
        return PromptHead(K, d, n_views, bc["prompt_len"], bc["prompt_layers"], bc["prompt_heads"], dropout, base_rate)
    if name == "shaspec":
        return ShaSpecHead(K, d, n_views, bc["shaspec_align"], bc["shaspec_domain"], base_rate)
    if name == "vulnsense":
        return VulnSenseHead(K, d, bc["lstm_hidden"], bc["max_opcodes"], base_rate)
    if name == "cmml":
        return CMMLHead(K, d, bc["cmml_rec"], bc["cmml_mutual"], base_rate)
    if name == "qmf":
        return QMFHead(K, d, n_views, base_rate)
    if name == "pdf":
        return PDFHead(K, d, n_views, base_rate)
    raise ValueError(name)
