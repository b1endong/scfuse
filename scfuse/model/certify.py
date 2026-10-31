import numpy as np
import torch


def selector_share(sel, mask, fnsel, fnmask):
    gs = torch.where(fnmask & (fnsel >= 0), fnsel, torch.full_like(fnsel, -2))
    verified = (sel.unsqueeze(-1) == gs.unsqueeze(1)).any(-1) & mask & (sel >= 0)
    n = mask.sum(1)
    share = torch.where(n > 0, verified.sum(1).float() / n.clamp_min(1), torch.zeros_like(n, dtype=torch.float))
    return share, verified


def selector_coverage(sel, mask, fnsel, fnmask):
    gs = fnmask & (fnsel >= 0)
    ds = sel.masked_fill(~mask | (sel < 0), -3)
    hit = (fnsel.unsqueeze(-1) == ds.unsqueeze(1)).any(-1) & gs
    return hit.sum(1).float() / gs.sum(1).clamp_min(1).float()


def _split(r, s, omega, m, Q):
    keep = np.ones(r.shape[1], bool)
    keep[list(Q)] = False
    g = m * omega
    rest = g[:, keep, None]
    return keep, (rest * r[:, keep]).sum(1), (rest * (r[:, keep] + s[:, keep])).sum(1)


def inclusion_bound(r, s, omega, m, a, W: float, Q) -> tuple[np.ndarray, np.ndarray]:
    r, s, omega, m = (np.asarray(x, np.float64) for x in (r, s, omega, m))
    keep, Rr, Nr = _split(r, s, omega, m, Q)
    gq = (m * omega)[:, ~keep, None]
    NQ = (gq * (r[:, ~keep] + s[:, ~keep])).sum(1)
    T_rest = W + Nr
    return NQ / (T_rest + NQ), (Rr + a * W) / T_rest


def interval(r, s, omega, m, a, W: float, Emax: float, Q, omega_bar=None) -> tuple[np.ndarray, np.ndarray]:
    r, s, omega, m = (np.asarray(x, np.float64) for x in (r, s, omega, m))
    keep, Rr, Nr = _split(r, s, omega, m, Q)
    bar = (m if omega_bar is None else np.asarray(omega_bar, np.float64))[:, ~keep].sum(1, keepdims=True)
    A, B, E = a * W + Rr, W + Nr, Emax * bar
    return A / (B + E), (A + E) / (B + E)


def certified(lo: np.ndarray, hi: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return (tau < lo) | (tau > hi)
