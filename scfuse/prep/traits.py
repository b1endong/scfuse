import argparse
import json
from pathlib import Path

import numpy as np

from scfuse.data import load_packed
from scfuse.labels import ANALYZERS, load_label_map

GROUPS = {"weak_train": ("weak", "train"), "weak_test": ("weak", "test"), "gold_solidifi": ("solidifi", "gold_test"),
          "gold_smartbugs": ("smartbugs", "gold_test")}


def rows_of(p, off: dict, k: str, i: int) -> np.ndarray:
    a0, n = off[f"{k}_0"][i], off[f"{k}_n"][i]
    return np.asarray(p.arr[k][a0:a0 + n])


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a):
        return float("nan")
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Low-quality traits of the packed data (no test labels)")
    ap.add_argument("--cache", default="data/scfuse")
    ap.add_argument("--propensity", default="results/data/propensity.json")
    ap.add_argument("--out", default="results/data/traits.json")
    ap.add_argument("--max-points", type=int, default=2000)
    args = ap.parse_args()
    lm = load_label_map()
    p = load_packed(args.cache, ["solidifi_unchecked_send_acc"])
    meta = p.meta
    off = {c: meta[c].to_numpy() for c in meta.columns if c.endswith("_0") or c.endswith("_n")}
    kinds, splits = meta.kind.to_numpy(), meta.split.to_numpy()
    hit = {v: [0, 0, 0] for v in ("S", "D")}
    recovery = {g: [] for g in GROUPS}
    multi, mapped = 0, 0
    for i in range(len(meta)):
        g = set(rows_of(p, off, "G_fnsel", i).tolist())
        for v in ("S", "D"):
            units = rows_of(p, off, f"{v}_unit", i)
            s = {int(x) for x in units[:, 2].tolist() if x >= 0} if len(units) else set()
            hit[v][0] += len(s & g)
            hit[v][1] += len(s)
            hit[v][2] += len(g)
            if v == "D" and g:
                for name, (kind, split) in GROUPS.items():
                    if kinds[i] == kind and splits[i] == split:
                        recovery[name].append(len(s & g) / len(g))
        mem = rows_of(p, off, "G_mem", i)
        if len(mem):
            _, counts = np.unique(mem[:, 0], return_counts=True)
            multi += int((counts > 1).sum())
            mapped += len(counts)
    weak = meta.kind.eq("weak").to_numpy()
    lifted = weak & (meta.m_D.to_numpy() > 0)
    out = {"selector": {v: {"precision": hit[v][0] / max(hit[v][1], 1), "recall": hit[v][0] / max(hit[v][2], 1)}
                        for v in hit},
           "recovery_ecdf": {k: np.sort(v)[np.linspace(0, len(v) - 1, min(len(v), args.max_points)).astype(int)].tolist()
                             if v else [] for k, v in recovery.items()},
           "recovery_n": {k: len(v) for k, v in recovery.items()},
           "multi_function_blocks": multi / max(mapped, 1),
           "source_available_weak": float(meta.m_S.to_numpy()[weak].mean()),
           "lift_available_weak": float(meta.m_D.to_numpy()[weak].mean())}
    if "D_fidelity" in meta.columns:
        q = np.stack(meta.q_D.to_numpy())
        out.update(lift_compiles_weak=float(q[lifted, 0].mean()), lift_fidelity_weak=float(q[lifted, 1].mean()),
                   below_half_selectors_weak=float((meta.D_coverage.to_numpy()[lifted] < 0.5).mean()),
                   matched_weak=float(meta.matched.to_numpy()[weak].mean()),
                   compiled_weak=float(meta.compiled.to_numpy()[weak].mean()))
    if Path(args.propensity).exists():
        prop = json.load(open(args.propensity))
        out["propensity"] = {k: prop[k] for k in ("e_auroc_oof", "e_calibration", "n_pool_codes") if k in prop}
    y = np.stack(meta.y_gold.to_numpy()).astype(float)
    anchor = meta.split.eq("gold_anchor").to_numpy()
    obs = {a: np.stack(meta[f"obs_{a}"].to_numpy()).astype(bool) for a in ANALYZERS}
    lab = {a: np.stack(meta[f"y_{a}"].to_numpy()).astype(float) for a in ANALYZERS}
    rates, pos_rate, kap = {}, {}, {}
    both = weak & obs["mythril"].any(1)
    for a in ANALYZERS:
        rates[a], pos_rate[a] = {}, {}
        for k, c in enumerate(lm.codes):
            o = anchor & obs[a][:, k]
            n1, n0 = int((o & (y[:, k] > 0)).sum()), int((o & (y[:, k] == 0)).sum())
            rates[a][c] = [float((lab[a][o & (y[:, k] > 0), k].sum() + 1) / (n1 + 2)),
                           float((lab[a][o & (y[:, k] == 0), k].sum() + 1) / (n0 + 2)), n1, n0]
            ob = both & obs[a][:, k]
            pos_rate[a][c] = float(lab[a][ob, k].mean()) if ob.any() else None
    for k, c in enumerate(lm.codes):
        jointly = both & obs["slither"][:, k] & obs["mythril"][:, k]
        kap[c] = kappa(lab["slither"][jointly, k], lab["mythril"][jointly, k]) if jointly.any() else None
    out["analyzers"] = {"anchor_rates": rates, "weak_positive_rate": pos_rate, "kappa": kap,
                        "mythril_subset": int(both.sum())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print({k: v for k, v in out.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
