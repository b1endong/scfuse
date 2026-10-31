import numpy as np
import pytest
import torch

from scfuse.model.certify import certified, inclusion_bound, interval
from scfuse.model.fusion import deferral, fuse, opinion

W, EMAX, J, K = 2.0, 20.0, 5, 9


def draw(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = EMAX * torch.rand(n, J, K, generator=g)
    s = EMAX * torch.rand(n, J, K, generator=g)
    omega = torch.rand(n, J, generator=g)
    m = (torch.rand(n, J, generator=g) > 0.3).float()
    a = torch.rand(K, generator=g).clamp(0.05, 0.95)
    return r, s, omega, m, a


def test_missing_source_is_neutral():
    r, s, omega, m, a = draw(256)
    m[:, 2] = 0
    P, u, _, _ = fuse(r, s, omega, m, a, W)
    r2 = r.clone()
    r2[:, 2] = torch.rand_like(r2[:, 2]) * EMAX
    P2, u2, _, _ = fuse(r2, s, omega, m, a, W)
    assert torch.allclose(P, P2) and torch.allclose(u, u2)


def test_no_evidence_gives_prior():
    r, s, omega, m, a = draw(8)
    P, u, _, _ = fuse(r, s, omega, torch.zeros_like(m), a, W)
    assert torch.allclose(P, a.expand_as(P)) and torch.allclose(u, torch.ones_like(u))


def test_prior_preserving_decomposition_and_floor():
    r, s, omega, m, a = draw(512, 1)
    P, u, R, S = fuse(r, s, omega, m, a, W)
    n = (m * omega).unsqueeze(-1) * (r + s)
    pbar = r / (r + s)
    T = W + n.sum(1)
    assert torch.allclose(P, u * a + (n / T.unsqueeze(1) * pbar).sum(1), atol=1e-5)
    assert torch.allclose(u + (n / T.unsqueeze(1)).sum(1), torch.ones_like(u), atol=1e-5)
    assert (u >= W / (W + 2 * J * EMAX) - 1e-7).all()


@pytest.mark.parametrize("seed", range(5))
def test_inclusion_bound_and_interval(seed):
    r, s, omega, m, a = draw(2000, seed)
    Q = (2, 4)
    P, _, _, _ = fuse(r, s, omega, m, a, W)
    delta, P_minus = inclusion_bound(r.numpy(), s.numpy(), omega.numpy(), m.numpy(), a.numpy(), W, Q)
    assert (np.abs(P.numpy() - P_minus) <= delta + 1e-6).all()
    lo, hi = interval(r.numpy(), s.numpy(), omega.numpy(), m.numpy(), a.numpy(), W, EMAX, Q)
    g = torch.Generator().manual_seed(100 + seed)
    for _ in range(20):
        r2, s2, om2 = r.clone(), s.clone(), omega.clone()
        for j in Q:
            r2[:, j] = EMAX * torch.rand(r.shape[0], K, generator=g)
            s2[:, j] = EMAX * torch.rand(r.shape[0], K, generator=g)
            om2[:, j] = torch.rand(r.shape[0], generator=g)
        P2, _, _, _ = fuse(r2, s2, om2, m, a, W)
        assert ((P2.numpy() >= lo - 1e-6) & (P2.numpy() <= hi + 1e-6)).all()
    tau = np.full(K, 0.5)
    cert = certified(lo, hi, tau)
    assert cert.dtype == bool and cert.shape == lo.shape


def test_opinion_and_deferral():
    r, s, omega, m, a = draw(64, 3)
    P, u = opinion(r, s, a, W)
    assert ((P > 0) & (P < 1)).all() and ((u > 0) & (u <= 1)).all()
    Pf, uf, _, _ = fuse(r, s, omega, m, a, W)
    ups = deferral(r, s, omega, m, uf, W, 0.5)
    assert (ups >= uf - 1e-7).all()
    assert torch.allclose(deferral(r, s, omega, m, uf, W, 0.0), uf)


def test_ds_rule_reduces_to_vacuous_for_missing_sources():
    r, s, omega, m, a = draw(32, 4)
    m[:] = 0
    m[:, 0] = 1
    P, u, _, _ = fuse(r, s, torch.ones_like(omega), m, a, W, "ds")
    P1, u1 = opinion(r[:, 0], s[:, 0], a, W)
    assert torch.allclose(P, P1, atol=1e-5) and torch.allclose(u, u1, atol=1e-5)
