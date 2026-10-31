import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from scfuse.utils.evm import N_OPCODES

N_ETYPES = 5
PREFIX = 3


def _tf(d, layers, heads, dropout):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class ViewAdapter(nn.Module):
    def __init__(self, d_in: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, n_views: int, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base, self.scale, self.drop, self.spans = base, alpha / r, nn.Dropout(dropout), None
        self.A = nn.Parameter(torch.empty(n_views, r, base.in_features))
        for v in range(n_views):
            nn.init.kaiming_uniform_(self.A.data[v], a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(n_views, base.out_features, r))

    def forward(self, x):
        x_d = self.drop(x)
        delta = torch.cat([x_d[a:b] @ self.A[v].T @ self.B[v].T for v, (a, b) in enumerate(self.spans)], 0)
        return self.base(x) + self.scale * delta


class LoraText(nn.Module):
    def __init__(self, backbone: str, n_views: int, d: int, r: int, alpha: float, dropout: float, grad_ckpt: bool,
                 max_rows: int):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.bb = AutoModel.from_pretrained(backbone, add_pooling_layer=False)
        self.bb.requires_grad_(False)
        for layer in self.bb.encoder.layer:
            att = layer.attention.self
            att.query = LoRALinear(att.query, n_views, r, alpha, dropout)
            att.value = LoRALinear(att.value, n_views, r, alpha, dropout)
        if grad_ckpt:
            self.bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        tok = AutoTokenizer.from_pretrained(backbone)
        self.register_buffer("prefix", torch.tensor([tok.cls_token_id, tok.convert_tokens_to_ids("<encoder-only>"),
                                                     tok.sep_token_id]), persistent=False)
        self.sep, self.pad, self.max_rows = tok.sep_token_id, tok.pad_token_id, max_rows
        self.proj = nn.ModuleList(nn.Linear(self.bb.config.hidden_size, d) for _ in range(n_views))
        self.loras = [m for m in self.bb.modules() if isinstance(m, LoRALinear)]

    def _ids(self, tok, n):
        C, L = tok.shape
        body = torch.where(torch.arange(L, device=tok.device) < n.unsqueeze(1), tok, self.pad)
        ids = torch.cat([self.prefix.expand(C, PREFIX), body, torch.full((C, 1), self.pad, device=tok.device)], 1)
        ids[torch.arange(C, device=tok.device), PREFIX + n] = self.sep
        return ids

    def _encode(self, ids, spans, sp=None):
        for mod in self.loras:
            mod.spans = spans
        mask = ids != self.pad
        h = self.bb(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        if sp is None or not len(sp[0]):
            return pooled, None
        r, lo, hi = sp
        cs = h.float().cumsum(1)
        top = h.shape[1] - 1
        span = cs[r, (hi + PREFIX - 1).clamp(0, top)] - cs[r, (lo + PREFIX - 1).clamp(0, top)]
        return pooled, span / (hi - lo).clamp_min(1).unsqueeze(1).float()

    def forward(self, views: list[dict], keep: list[torch.Tensor], span: dict | None = None):
        rows = [(self._ids(t["tok"][k], t["tok_len"][k]), t["chunk_unit"][k]) for t, k in zip(views, keep)]
        L = max([ids.shape[1] for ids, _ in rows] + [PREFIX + 1])
        ids = torch.cat([F.pad(x, (0, L - x.shape[1]), value=self.pad) for x, _ in rows], 0)
        bounds = [0]
        for x, _ in rows:
            bounds.append(bounds[-1] + len(x))
        sp, ok = None, None
        if span is not None:
            c = span["chunk"].clamp_min(0)
            ok = span["valid"] & keep[0][c]
            sp = ((keep[0].long().cumsum(0) - 1)[c][ok], span["lo"][ok], span["hi"][ok])
        H = self.bb.config.hidden_size
        if torch.is_grad_enabled():
            pooled, span_h = self._encode(ids, list(zip(bounds[:-1], bounds[1:])), sp)
        else:
            parts, span_parts = [], []
            for s in range(0, len(ids), self.max_rows):
                e = min(s + self.max_rows, len(ids))
                clip = lambda i: min(max(i, s), e) - s
                local = None
                if sp is not None:
                    inside = (sp[0] >= s) & (sp[0] < e)
                    local = (sp[0][inside] - s, sp[1][inside], sp[2][inside])
                p, sh = self._encode(ids[s:e], [(clip(a), clip(b)) for a, b in zip(bounds[:-1], bounds[1:])], local)
                parts.append(p)
                if sh is not None:
                    span_parts.append(sh)
            pooled = torch.cat(parts, 0) if parts else ids.new_zeros((0, H)).float()
            span_h = torch.cat(span_parts, 0) if span_parts else None
        out = []
        for v, (t, (_, cu)) in enumerate(zip(views, rows)):
            B, U = t["mask"].shape
            p = pooled[bounds[v]:bounds[v + 1]].float()
            s = p.new_zeros(B * U, p.shape[1]).index_add_(0, cu, p)
            n = p.new_zeros(B * U).index_add_(0, cu, torch.ones_like(cu, dtype=p.dtype)).clamp_min(1)
            out.append(self.proj[v]((s / n.unsqueeze(1)).view(B, U, -1)))
        spans_out = None
        if span is not None:
            spans_out = pooled.new_zeros(len(ok), self.proj[0].out_features)
            if span_h is not None:
                spans_out[ok] = self.proj[0](span_h).to(spans_out.dtype)
        return out, spans_out, ok


class OpcodeEncoder(nn.Module):
    def __init__(self, d: int = 128, layers: int = 2, heads: int = 4, dropout: float = 0.1, max_len: int = 32):
        super().__init__()
        self.mask_id = N_OPCODES + 1
        self.emb = nn.Embedding(N_OPCODES + 2, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.tf = _tf(d, layers, heads, dropout)
        self.norm = nn.LayerNorm(d)

    def tokens(self, x):
        pad = x == 0
        pad[:, 0] = False
        return self.norm(self.tf(self.emb(x) + self.pos[:x.shape[1]], src_key_padding_mask=pad))

    def forward(self, x):
        keep = (x != 0).unsqueeze(-1)
        return (self.tokens(x) * keep).sum(1) / keep.sum(1).clamp_min(1)


@torch.no_grad()
def rwse(adj: torch.Tensor, steps: int) -> torch.Tensor:
    A = ((adj > 0) | (adj > 0).transpose(1, 2)).float()
    P = A / A.sum(-1, keepdim=True).clamp_min(1)
    out, M = [], P
    with torch.autocast(device_type=adj.device.type, enabled=False):
        for _ in range(steps):
            out.append(torch.diagonal(M, dim1=1, dim2=2))
            M = M @ P
    return torch.stack(out, -1)


class _BiasedLayer(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.p = heads, dropout
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias):
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype),
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, d)))
        return x + self.drop(self.ff(self.n2(x)))


class GraphEncoder(nn.Module):
    def __init__(self, d_block: int, d: int, layers: int, heads: int, rw_steps: int, dropout: float):
        super().__init__()
        self.rw_steps = rw_steps
        self.inp, self.rw = nn.Linear(d_block, d), nn.Linear(rw_steps, d)
        self.deg_in, self.deg_out = nn.Embedding(17, d), nn.Embedding(17, d)
        self.ebias, self.ebias_rev = nn.Embedding(N_ETYPES + 1, heads), nn.Embedding(N_ETYPES + 1, heads)
        self.layers = nn.ModuleList(_BiasedLayer(d, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, blocks, adj, mask):
        A = (adj > 0).float()
        h = (self.inp(blocks) + self.rw(rwse(adj, self.rw_steps))
             + self.deg_out(A.sum(2).clamp(max=16).long()) + self.deg_in(A.sum(1).clamp(max=16).long()))
        a = adj.long()
        bias = (self.ebias(a) + self.ebias_rev(a.transpose(1, 2))).permute(0, 3, 1, 2)
        key = mask | ~mask.any(1, keepdim=True)
        bias = bias.masked_fill(~key[:, None, None, :], float("-inf"))
        for layer in self.layers:
            h = layer(h, bias)
        return self.norm(h)


class AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h, member):
        s = self.score(h).squeeze(-1).float().unsqueeze(1).masked_fill(~member, -1e9)
        w = torch.softmax(s, -1) * member
        return torch.einsum("bfn,bnd->bfd", w.to(h.dtype), h)


class ASTEncoder(nn.Module):
    def __init__(self, vocab: int, d_ast: int, d: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_ast, padding_idx=0)
        self.depth, self.rank = nn.Embedding(64, d_ast), nn.Embedding(16, d_ast)
        self.tf = _tf(d_ast, layers, heads, dropout)
        self.out = nn.Sequential(nn.LayerNorm(d_ast), nn.Linear(d_ast, d))

    def forward(self, node):
        pad = node[..., 0] == 0
        pad[:, 0] = False
        x = self.emb(node[..., 0]) + self.depth(node[..., 1].clamp(0, 63)) + self.rank(node[..., 2].clamp(0, 15))
        h = self.tf(x, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        return self.out((h * keep).sum(1) / keep.sum(1).clamp_min(1))


class CrossView(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, dropout: float, n_views: int):
        super().__init__()
        self.vtype = nn.Embedding(n_views, d)
        self.tf = _tf(d, layers, heads, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, hs, masks):
        x = torch.cat([h + self.vtype.weight[i].to(h.dtype) for i, h in enumerate(hs)], 1)
        keep = torch.cat(masks, 1)
        safe = keep.clone()
        safe[:, 0] |= ~keep.any(1)
        return self.norm(self.tf(x, src_key_padding_mask=~safe)), keep


class LabelAttention(nn.Module):
    def __init__(self, K: int, d: int):
        super().__init__()
        self.g = nn.Parameter(torch.randn(K, d) * d ** -0.5)

    def forward(self, h, mask):
        logits = torch.einsum("kd,bud->bku", self.g.to(h.dtype), h).float() / math.sqrt(h.shape[-1])
        w = torch.softmax(logits.masked_fill(~mask.unsqueeze(1), -1e9), -1) * mask.unsqueeze(1)
        return torch.einsum("bku,bud->bkd", w.to(h.dtype), h), w
