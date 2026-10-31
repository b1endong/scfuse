import argparse
import json
from pathlib import Path

import numpy as np

from scfuse.config import load_config
from scfuse.labels import load_label_map
from scfuse.metrics import auroc, f1_per_class


def resample_counts(n_items: int, n: int, seed: int, groups: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if groups is None:
        return rng.multinomial(n_items, np.full(n_items, 1 / n_items), size=n).astype(np.float32)
    uniq, inv = np.unique(groups, return_inverse=True)
    G = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n).astype(np.float32)
    return G[:, inv]


def resampled_macro_f1(preds: list, y: np.ndarray, idx: list, Wt: np.ndarray) -> np.ndarray:
    t = y[:, idx].astype(np.float32)
    pos = Wt @ t
    vals = []
    for pr in preds:
        p = pr[:, idx].astype(np.float32)
        tp, fp, fn = Wt @ (p * t), Wt @ (p * (1 - t)), Wt @ ((1 - p) * t)
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(pos > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        vals.append(np.nanmean(f1, 1))
    return np.mean(vals, 0)


def point_macro_f1(preds: list, y: np.ndarray, idx: list) -> float:
    with np.errstate(all="ignore"):
        return float(np.mean([np.nanmean(f1_per_class(p, y)[idx]) for p in preds]))


def bootstrap_ci(preds: list, y: np.ndarray, idx, n: int = 10_000, seed: int = 0, alpha: float = 0.05,
                 groups=None) -> tuple[float, float]:
    m = resampled_macro_f1(preds, y, list(idx), resample_counts(len(y), n, seed, groups))
    return float(np.nanpercentile(m, 100 * alpha / 2)), float(np.nanpercentile(m, 100 * (1 - alpha / 2)))


def paired(preds_ref: list, preds_other: list, y: np.ndarray, idx, n: int = 10_000, seed: int = 0,
           alpha: float = 0.05, groups=None) -> dict:
    idx = list(idx)
    Wt = resample_counts(len(y), n, seed, groups)
    d = resampled_macro_f1(preds_other, y, idx, Wt) - resampled_macro_f1(preds_ref, y, idx, Wt)
    return {"delta": point_macro_f1(preds_other, y, idx) - point_macro_f1(preds_ref, y, idx),
            "lo": float(np.nanpercentile(d, 100 * alpha / 2)), "hi": float(np.nanpercentile(d, 100 * (1 - alpha / 2))),
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def macro_auroc(scores: list, L: np.ndarray, classes: list) -> float:
    vals = []
    for k in classes:
        known = ~np.isnan(L[:, k])
        a = np.mean([auroc(s[known, k], L[known, k] > 0) for s in scores])
        if not np.isnan(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else float("nan")


def project_bootstrap(scores_a: list, scores_b: list | None, L: np.ndarray, classes: list, projects: np.ndarray,
                      n: int = 10_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(projects)
    members = [np.flatnonzero(projects == p) for p in uniq]
    vals = []
    for _ in range(n):
        rows = np.concatenate([members[i] for i in rng.integers(len(uniq), size=len(uniq))])
        a = macro_auroc([s[rows] for s in scores_a], L[rows], classes)
        vals.append(a - macro_auroc([s[rows] for s in scores_b], L[rows], classes) if scores_b is not None else a)
    vals = np.asarray(vals)
    point = macro_auroc(scores_a, L, classes) - (macro_auroc(scores_b, L, classes) if scores_b is not None else 0)
    return {"point": point, "lo": float(np.nanpercentile(vals, 2.5)), "hi": float(np.nanpercentile(vals, 97.5))}


def mde(preds: list, y: np.ndarray, idx, power_z: float = 0.84, alpha_z: float = 1.96, n: int = 2000) -> float:
    m = resampled_macro_f1(preds, y, list(idx), resample_counts(len(y), n, 1))
    return float((alpha_z + power_z) * np.sqrt(2) * np.nanstd(m))


def load_preds(eval_dir: Path, run: str, key: str, seeds: list | None) -> tuple[list, dict]:
    files = sorted((eval_dir / run).glob("pred_s*.npz"))
    out, meta = [], None
    for f in files:
        if seeds and int(f.stem.split("_s")[1]) not in seeds:
            continue
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        x = z[key]
        out.extend(list(x) if x.ndim == 3 else [x])
        meta = {"y": z["y"], "kind": z["kind"], "host": z["host"] if "host" in z.files else None}
    return out, meta


def compare(cfg: dict, ref: str, runs: list, protocol: str, subset: str, seeds: list | None, groups: str) -> dict:
    lm = load_label_map()
    eval_dir = Path(cfg["run"]["eval"])
    key = f"{protocol}_pred"
    ref_preds, meta = load_preds(eval_dir, ref, key, seeds)
    if not ref_preds:
        raise SystemExit(f"no {key} for {ref} in {eval_dir}")
    rows = np.ones(len(meta["y"]), bool) if subset == "all" else meta["kind"] == subset
    g = meta["host"][rows] if groups == "host" and meta["host"] is not None else None
    res = {}
    for run in runs:
        other, _ = load_preds(eval_dir, run, key, None)
        if other:
            res[run] = paired([p[rows] for p in ref_preds], [p[rows] for p in other], meta["y"][rows], lm.eval_idx,
                              groups=g)
            res[run]["n_sets"] = len(other)
    adj = holm([r["p"] for r in res.values()])
    for r, a in zip(res.values(), adj):
        r["p_holm"] = float(a)
    return {"ref": ref, "protocol": protocol, "subset": subset, "n_ref_sets": len(ref_preds), "runs": res}


def main():
    ap = argparse.ArgumentParser(description="Paired bootstrap of gold-test macro-F1 against a reference method")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ref", default="scfuse")
    ap.add_argument("--protocol", default="P1")
    ap.add_argument("--subset", default="all", choices=("all", "solidifi", "smartbugs"))
    ap.add_argument("--groups", default="contract", choices=("contract", "host"))
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = compare(cfg, args.ref, args.runs, args.protocol, args.subset, args.seeds, args.groups)
    out = Path(cfg["run"]["analysis"]) / f"paired_{args.ref}_{args.protocol}_{args.subset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    for run, r in res["runs"].items():
        print(f"{run:24s} delta {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] p_holm {r['p_holm']:.3f}")
    print("->", out)


if __name__ == "__main__":
    main()
