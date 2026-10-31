import math

import torch

from scfuse.model.losses import (AnalyzerNoise, anchor_rates, caa_function_terms, caa_span_terms, evidential_loss,
                                  multi_positive_nce, weak_loss)


def test_multi_positive_nce_lemma1_lower_bound():
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(6, 16, generator=g), torch.randn(10, 16, generator=g)
    pos = torch.zeros(6, 10, dtype=torch.bool)
    pos[torch.arange(6), torch.arange(6)] = True
    pos[0, 7] = True
    cand = torch.ones(6, 10, dtype=torch.bool)
    t = 0.07
    loss = multi_positive_nce(q, k, pos, cand, t)
    n_pos, n_neg = pos.sum(1).float(), (~pos).sum(1).float()
    assert (loss >= torch.log1p(n_neg / n_pos * math.exp(-2 / t)) - 1e-5).all()


def test_function_terms_skip_masked_views():
    B, U, d = 2, 3, 8
    h = [torch.randn(B, U, d) for _ in range(4)]
    sel = [torch.tensor([[1, 2, -1], [3, 4, 5]]) for _ in range(4)]
    mask = [torch.ones(B, U, dtype=torch.bool) for _ in range(4)]
    m = torch.ones(B, 4)
    m[:, 2] = 0
    terms = caa_function_terms(h, sel, mask, m, [(0, 1), (0, 2)], 0.07)
    assert len(terms) == 2 and all(torch.isfinite(x).all() for x in terms)


def test_span_terms_exclude_overlaps():
    h_block, h_span = torch.randn(4, 8), torch.randn(3, 8)
    span_of = torch.tensor([0, 0, 1, 2])
    b, unit = torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0])
    lo, hi = torch.tensor([0, 5, 0]), torch.tensor([10, 20, 4])
    terms = caa_span_terms(h_block, h_span, span_of, b, unit, lo, hi, torch.ones(3, dtype=torch.bool), 0.07)
    assert all(torch.isfinite(x).all() for x in terms)


def test_evidential_loss_finite_and_weighted():
    R, S = 20 * torch.rand(4, 9), 20 * torch.rand(4, 9)
    y = (torch.rand(4, 9) > 0.5).float()
    a = torch.full((9,), 0.3)
    full = evidential_loss(R, S, y, a, 2.0, 1.0)
    half = evidential_loss(R, S, y, a, 2.0, 1.0, torch.full((4, 9), 0.5))
    assert torch.isfinite(full).all() and torch.allclose(half, full / 2)


def test_noise_parametrization_orders_rates():
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y_an = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]], [[1.0, 1.0]], [[0.0, 0.0]]])
    obs = torch.ones_like(y_an)
    alpha, beta, use = anchor_rates(y, torch.ones_like(y), y_an, obs, 1.0, 1)
    noise = AnalyzerNoise(alpha, beta, use, 20.0)
    a2, b2 = noise.rates()
    assert (a2 > b2).all() and torch.allclose(a2[use], alpha[use], atol=1e-3)
    P = torch.rand(4, 2)
    assert torch.isfinite(weak_loss(P, y_an, obs, a2, b2)).all()
    assert torch.isfinite(noise.prior_nll())
