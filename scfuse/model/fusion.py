import torch

RULES = ("cumulative", "average", "ds")


def opinion(r, s, a, W: float):
    return (r + a * W) / (r + s + W), W / (r + s + W)


def fuse(r, s, omega, m, a, W: float = 2.0, rule: str = "cumulative"):
    g = (m * omega).unsqueeze(-1)
    er, es = g * r, g * s
    if rule == "cumulative":
        R, S = er.sum(1), es.sum(1)
    elif rule == "average":
        n = m.sum(1).clamp_min(1).unsqueeze(-1)
        R, S = er.sum(1) / n, es.sum(1) / n
    elif rule == "ds":
        tot = er + es + W
        b, d, u = er / tot, es / tot, W / tot
        B, D, U = b[:, 0], d[:, 0], u[:, 0]
        for j in range(1, r.shape[1]):
            conflict = B * d[:, j] + D * b[:, j]
            norm = (1 - conflict).clamp_min(1e-12)
            B, D, U = ((B * b[:, j] + B * u[:, j] + b[:, j] * U) / norm,
                       (D * d[:, j] + D * u[:, j] + d[:, j] * U) / norm, U * u[:, j] / norm)
        return B + a * U, U, B * W / U, D * W / U
    else:
        raise ValueError(rule)
    P, u = opinion(R, S, a, W)
    return P, u, R, S


def mass(r, s, omega, m):
    return (m * omega).unsqueeze(-1) * (r + s)


def deferral(r, s, omega, m, u, W: float, lam_dis: float):
    n = mass(r, s, omega, m)
    c = n / (n + W)
    p = r / (r + s).clamp_min(1e-12)
    J = r.shape[1]
    iu = torch.triu_indices(J, J, 1, device=r.device)
    cc = c[:, iu[0]] * c[:, iu[1]]
    num = (cc * (p[:, iu[0]] - p[:, iu[1]]).abs()).sum(1)
    den = cc.sum(1)
    two = ((n > 0).sum(1) >= 2) & (den > 0)
    return u + lam_dis * torch.where(two, num / den.clamp_min(1e-12), torch.zeros_like(num))
