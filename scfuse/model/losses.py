import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_beta(a1, b1, a2, b2):
    return (torch.lgamma(a1 + b1) - torch.lgamma(a1) - torch.lgamma(b1)
            - torch.lgamma(a2 + b2) + torch.lgamma(a2) + torch.lgamma(b2)
            + (a1 - a2) * torch.digamma(a1) + (b1 - b2) * torch.digamma(b1)
            + (a2 - a1 + b2 - b1) * torch.digamma(a1 + b1))


def evidential_loss(R, S, y, a, W: float, lam_kl: float, weight=None):
    gp, gm = R + a * W, S + (1 - a) * W
    tot = torch.digamma(gp + gm)
    nll = y * (tot - torch.digamma(gp)) + (1 - y) * (tot - torch.digamma(gm))
    hp = a * W + (1 - y) * R
    hm = (1 - a) * W + y * S
    out = nll + lam_kl * kl_beta(hp, hm, a * W, (1 - a) * W)
    return (out if weight is None else out * weight).sum(-1)


def reliability_loss(omega_logit, m, Pj, y, weight):
    target = 1 - (weight.unsqueeze(1) * (y.unsqueeze(1) - Pj).abs()).sum(-1) / weight.sum(-1, keepdim=True)
    bce = F.binary_cross_entropy_with_logits(omega_logit, target.detach().clamp(0, 1), reduction="none")
    return (m * bce).sum(1), m.sum(1)


def weak_loss(P, y_an, obs, alpha, beta, eps: float = 1e-6):
    Pt = (beta + (alpha - beta) * P.unsqueeze(1)).clamp(eps, 1 - eps)
    return -(obs * (y_an * torch.log(Pt) + (1 - y_an) * torch.log(1 - Pt))).sum((1, 2))


def majority_loss(P, y_an, obs, eps: float = 1e-6):
    seen = obs.amax(1)
    target = (y_an * obs).amax(1)
    P = P.clamp(eps, 1 - eps)
    return -(seen * (target * P.log() + (1 - target) * (1 - P).log())).sum(-1)


def anchor_rates(y, weight, y_an, obs, eps0: float, min_pos: int, fallback=None):
    yw = (y * weight).unsqueeze(1)
    nw = ((1 - y) * weight).unsqueeze(1)
    n1, n0 = (obs * yw).sum(0), (obs * nw).sum(0)
    alpha = ((obs * yw * y_an).sum(0) + eps0) / (n1 + 2 * eps0)
    beta = ((obs * nw * y_an).sum(0) + eps0) / (n0 + 2 * eps0)
    pos = (obs * y.unsqueeze(1)).sum(0)
    use = (pos >= min_pos) & (n0 > 0) & (alpha > beta)
    if fallback is not None:
        none = (n1 + n0) == 0
        alpha = torch.where(none, torch.full_like(alpha, fallback[0]), alpha)
        beta = torch.where(none, torch.full_like(beta, fallback[1]), beta)
        use = use | none
    return alpha, beta, use


class AnalyzerNoise(nn.Module):
    def __init__(self, alpha0, beta0, use, kappa: float):
        super().__init__()
        alpha0, beta0 = alpha0.float(), torch.minimum(beta0.float(), alpha0.float() - 1e-3).clamp_min(1e-3)
        self.register_buffer("alpha0", alpha0)
        self.register_buffer("beta0", beta0)
        self.register_buffer("use", use.float())
        self.kappa = kappa
        self.nb = nn.Parameter(torch.logit(beta0))
        self.na = nn.Parameter(torch.logit(((alpha0 - beta0) / (1 - beta0)).clamp(1e-3, 1 - 1e-3)))

    def rates(self):
        beta = torch.sigmoid(self.nb)
        return beta + (1 - beta) * torch.sigmoid(self.na), beta

    def prior_nll(self):
        alpha, beta = self.rates()
        out = 0
        for rate, r0 in ((alpha, self.alpha0), (beta, self.beta0)):
            rate = rate.clamp(1e-6, 1 - 1e-6)
            out = out - (self.use * self.kappa * (r0 * rate.log() + (1 - r0) * (1 - rate).log())).sum()
        return out

    def forward(self, P, y_an, obs, detach_rates: bool):
        alpha, beta = self.rates()
        if detach_rates:
            alpha, beta = alpha.detach(), beta.detach()
        return weak_loss(P, y_an, obs * self.use, alpha, beta)


def multi_positive_nce(q, k, pos, cand, t: float):
    logits = F.normalize(q.float(), dim=-1) @ F.normalize(k.float(), dim=-1).T / t
    has = pos.any(1)
    if not has.any():
        return logits.new_zeros(0)
    num = torch.logsumexp(logits.masked_fill(~pos, -1e9), 1)
    den = torch.logsumexp(logits.masked_fill(~(cand | pos), -1e9), 1)
    return (den - num)[has]


def caa_function_terms(h, sel, mask, m, pairs, t: float):
    terms = []
    for v, w in pairs:
        okv = mask[v] & (m[:, v:v + 1] > 0)
        okw = mask[w] & (m[:, w:w + 1] > 0)
        bv, iv = okv.nonzero(as_tuple=True)
        bw, iw = okw.nonzero(as_tuple=True)
        if not len(bv) or not len(bw):
            continue
        sv, sw = sel[v][bv, iv], sel[w][bw, iw]
        pos = (bv[:, None] == bw[None]) & (sv[:, None] == sw[None]) & (sv[:, None] >= 0)
        cand = torch.ones_like(pos)
        hv, hw = h[v][bv, iv], h[w][bw, iw]
        terms.append(multi_positive_nce(hv, hw, pos, cand, t))
        terms.append(multi_positive_nce(hw, hv, pos.T, cand.T, t))
    return terms


def caa_span_terms(h_block, h_span, span_of, span_b, span_unit, lo, hi, valid, t: float):
    ok = valid[span_of]
    if not ok.any():
        return []
    h_block, span_of = h_block[ok], span_of[ok]
    Q = len(h_span)
    pos = torch.zeros(len(span_of), Q, dtype=torch.bool, device=h_span.device)
    pos[torch.arange(len(span_of), device=h_span.device), span_of] = True
    same = (span_b[:, None] == span_b[None]) & (span_unit[:, None] == span_unit[None])
    overlap = same & (lo[:, None] < hi[None]) & (lo[None] < hi[:, None])
    cand_span = ~overlap[span_of] & valid[None]
    terms = [multi_positive_nce(h_block, h_span, pos, cand_span, t)]
    blocks_overlap = overlap[:, span_of]
    terms.append(multi_positive_nce(h_span, h_block, pos.T, ~blocks_overlap, t))
    return terms


def caa_contract_loss(h, mask, m, pairs, t: float):
    pooled = [(x * k.unsqueeze(-1)).sum(1) / k.sum(1, keepdim=True).clamp_min(1) for x, k in zip(h, mask)]
    total = h[0].new_zeros((), dtype=torch.float32)
    for v, w in pairs:
        b = ((m[:, v] > 0) & (m[:, w] > 0) & mask[v].any(1) & mask[w].any(1)).nonzero(as_tuple=True)[0]
        if len(b) > 1:
            eye = torch.eye(len(b), dtype=torch.bool, device=b.device)
            total = total + 0.5 * (multi_positive_nce(pooled[v][b], pooled[w][b], eye, ~eye, t).mean()
                                   + multi_positive_nce(pooled[w][b], pooled[v][b], eye, ~eye, t).mean())
    return total


def conflict_loss(r, s, m, W: float):
    p, u = r / (r + s + W), W / (r + s + W)
    J = r.shape[1]
    iu = torch.triu_indices(J, J, 1, device=r.device)
    both = (m[:, iu[0]] * m[:, iu[1]]).unsqueeze(-1)
    deg = (p[:, iu[0]] - p[:, iu[1]]).abs() * (1 - u[:, iu[0]]) * (1 - u[:, iu[1]])
    return (both * deg).sum((1, 2)) / both.sum((1, 2)).clamp_min(1)


class TransitionNoise(nn.Module):
    def __init__(self, rows, n_dataset: int, n_views: int, K: int):
        super().__init__()
        self.V, self.K = n_views, K
        row_of = torch.full((n_dataset,), -1, dtype=torch.long)
        row_of[torch.as_tensor(rows, dtype=torch.long)] = torch.arange(len(rows))
        self.register_buffer("row_of", row_of)
        self.T = nn.Embedding(len(rows), n_views * K * 4, sparse=True)
        with torch.no_grad():
            self.T.weight.copy_(5.0 * torch.eye(2).flatten().repeat(n_views * K))

    def transfer(self, idx, r, s):
        rows = self.row_of[idx]
        assert (rows >= 0).all(), "transition requested for a contract outside training"
        T = torch.softmax(self.T(rows).view(len(idx), self.V, self.K, 2, 2).float(), -1)
        e2 = torch.einsum("bvki,bvkij->bvkj", torch.stack([r[:, :self.V], s[:, :self.V]], -1).float(), T)
        return e2[..., 0], e2[..., 1], T


def transition_regularizers(T, conf, t, keep, m_view):
    lab = t.long().unsqueeze(1).expand_as(conf)
    idx = 1 - lab
    diag = torch.diagonal(T, dim1=-2, dim2=-1)
    d_true = diag.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    d_other = diag.gather(-1, (1 - idx).unsqueeze(-1)).squeeze(-1)
    w = keep.unsqueeze(1) * m_view.unsqueeze(-1)
    mean_conf = []
    for c in (0, 1):
        sel = w * (lab == c)
        mean_conf.append((conf * sel).sum(0) / sel.sum(0).clamp_min(1))
    other = torch.where(lab == 1, mean_conf[0].unsqueeze(0), mean_conf[1].unsqueeze(0))
    l_conf = (w * ((conf - d_true) ** 2 + (other - d_other) ** 2)).sum() / w.sum().clamp_min(1)
    l_cons = T.new_zeros(())
    for v in range(T.shape[1]):
        for u in range(v + 1, T.shape[1]):
            both = (m_view[:, v] * m_view[:, u]).unsqueeze(-1)
            l_cons = l_cons + (both * (T[:, v] - T[:, u]).pow(2).mean((-2, -1))).sum() / both.sum().clamp_min(1) / T.shape[2]
    return l_conf, l_cons
