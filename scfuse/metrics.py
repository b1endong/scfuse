import numpy as np

GRID = np.linspace(0.01, 0.99, 99)
FINE_GRID = np.unique(np.concatenate([np.geomspace(1e-4, 1e-2, 21), GRID]))
UMAX_GRID = np.round(np.linspace(0.01, 1.0, 100), 3)


def f1_per_class(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred, y = pred.astype(bool), y.astype(bool)
    tp, fp, fn = (pred & y).sum(0), (pred & ~y).sum(0), (~pred & y).sum(0)
    return np.where(y.sum(0) > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)


def tune_thresholds(P: np.ndarray, y: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    pred = P[None] >= grid[:, None, None]
    yb = y.astype(bool)[None]
    tp, fp, fn = (pred & yb).sum(1), (pred & ~yb).sum(1), (~pred & yb).sum(1)
    tau = grid[np.argmax(2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)]
    return np.where(y.sum(0) > 0, tau, 0.5)


def macro_f1(pred: np.ndarray, y: np.ndarray, idx) -> float:
    with np.errstate(all="ignore"):
        return float(np.nanmean(f1_per_class(pred, y)[list(idx)]))


def micro_f1(pred: np.ndarray, y: np.ndarray, idx) -> float:
    p, t = pred[:, list(idx)].astype(bool), y[:, list(idx)].astype(bool)
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def no_skill_f1(y: np.ndarray, idx) -> dict:
    p = y[:, list(idx)].mean(0)
    p = p[p > 0]
    return {"prevalence": round(float(p.mean()), 4), "all_positive": round(float((2 * p / (1 + p)).mean()), 4)}


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    lab = label[order].astype(float)
    if lab.sum() == 0:
        return float("nan")
    prec = np.cumsum(lab) / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / lab.sum())


def auprc(P: np.ndarray, y: np.ndarray, idx) -> float:
    ap = [average_precision(P[:, k], y[:, k]) for k in idx if y[:, k].any()]
    return float(np.mean(ap)) if ap else float("nan")


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    label = label.astype(bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_auroc(score: np.ndarray, label: np.ndarray, strata: np.ndarray) -> float:
    vals, weights = [], []
    for s in np.unique(strata):
        k = strata == s
        a = auroc(score[k], label[k])
        if not np.isnan(a):
            vals.append(a)
            weights.append(label[k].sum())
    return float(np.average(vals, weights=weights)) if vals else float("nan")


def ece(P: np.ndarray, y: np.ndarray, idx=None, bins: int = 15) -> float:
    cols = list(range(P.shape[1])) if idx is None else list(idx)
    p, t = P[:, cols].ravel(), y[:, cols].ravel().astype(float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == i).mean() * abs(t[b == i].mean() - p[b == i].mean()) for i in range(bins) if (b == i).any()))


def selective(conf: np.ndarray, pred: np.ndarray, y: np.ndarray, idx, coverage: float = 0.8) -> tuple[float, float, np.ndarray]:
    err = (pred[:, list(idx)] != y[:, list(idx)].astype(bool)).mean(1)[np.argsort(-conf, kind="stable")]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean()), float(risk[max(1, int(round(coverage * len(err)))) - 1]), risk


def deferral(upsilon: np.ndarray, pred: np.ndarray, y: np.ndarray, idx, u_max: float) -> dict:
    keep = upsilon[:, list(idx)] <= u_max
    err = pred[:, list(idx)] != y[:, list(idx)].astype(bool)
    return {"coverage": float(keep.mean()), "selective_risk": float(err[keep].mean()) if keep.any() else float("nan"),
            "risk_full": float(err.mean())}


def select_umax(upsilon: np.ndarray, pred: np.ndarray, y: np.ndarray, idx, min_coverage: float,
                floor: float = 0.0) -> float:
    best, best_risk = 1.0, np.inf
    for u in UMAX_GRID[UMAX_GRID >= floor]:
        d = deferral(upsilon, pred, y, idx, u)
        if d["coverage"] >= min_coverage and d["selective_risk"] < best_risk:
            best, best_risk = float(u), d["selective_risk"]
    return best


def uncertainty_quintiles(u: np.ndarray, pred: np.ndarray, y: np.ndarray, idx, q: int = 5) -> list[float]:
    err = (pred[:, list(idx)] != y[:, list(idx)].astype(bool)).mean(1)
    order = np.argsort(u, kind="stable")
    return [float(err[part].mean()) for part in np.array_split(order, q)]
