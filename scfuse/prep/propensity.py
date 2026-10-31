import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scfuse.prep.common import Raw, Records, load_index, parser, pool_map
from scfuse.prep.index import sanctuary
from scfuse.utils.evm import COVARIATES, covariates, runtime

PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8,
          "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1}


def _key(code: str) -> str:
    return hashlib.sha1(runtime(code)).hexdigest()


def _z(code: str) -> np.ndarray:
    return covariates(code)


def population(raw: Raw, workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    pool = pd.read_parquet(raw.pool).sort_values(["block_timestamp", "address"], kind="stable")
    pool["address"] = pool["address"].str.lower()
    pool["key"] = pool["bytecode"].map(_key)
    pop = pool.groupby("key", sort=False).agg(address=("address", "first"), bytecode=("bytecode", "first"),
                                              n_deploy=("address", "size"), addresses=("address", list))
    Z = np.stack(list(pool_map(_z, pop["bytecode"].tolist(), workers, "covariates", 256, ordered=True)))
    keep = (Z[:, COVARIATES.index("eip1167")] == 0) & (Z[:, COVARIATES.index("log_n_selectors")] > 0)
    return pop[keep].drop(columns="bytecode").reset_index(drop=True), Z[keep]


def crossfit(X: np.ndarray, y: np.ndarray, eps: float, seed: int = 0):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        oof[te] = lgb.LGBMClassifier(random_state=seed, **PARAMS).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    model = lgb.LGBMClassifier(random_state=seed, **PARAMS).fit(X, y)
    return np.clip(oof, eps, 1 - eps), model


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [{"p": float(p[b == i].mean()), "rate": float(y[b == i].mean()), "n": int((b == i).sum())}
            for i in range(bins) if (b == i).any()]


def main():
    ap = parser("Verification propensity e(z) and density ratio w(z) from deployment bytecode")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--report", default="results/data/propensity.json")
    args = ap.parse_args()
    raw, cache = Raw(args.raw), Path(args.cache)
    pop, Zp = population(raw, args.workers)
    paths, meta = sanctuary(raw)
    mirror = set(paths) | set(meta)
    eth = {}
    if raw.etherscan.exists():
        for r in map(json.loads, open(raw.etherscan)):
            if r.get("status") == "ok":
                eth[r["address"].lower()] = bool(r["verified"])
    in_mirror = pop["addresses"].map(lambda a: any(x in mirror for x in a)).to_numpy()
    on_eth = pop["address"].map(eth)
    known = in_mirror | on_eth.notna().to_numpy()
    yv = (in_mirror | on_eth.eq(True).to_numpy()).astype(int)
    e_oof, e_model = crossfit(Zp[known], yv[known], args.eps)

    cfg = Records(cache / "cfg.jsonl")
    rows = [r for r in load_index(cache) if r["id"] in cfg]
    ids, Z, train = [], [], []
    for r in rows:
        rec = cfg.get(r["id"])
        if rec.get("z"):
            ids.append(r["id"])
            Z.append(rec["z"])
            train.append(r["split"] == "train")
    Z, train = np.asarray(Z, np.float32), np.asarray(train)
    e = np.clip(e_model.predict_proba(Z)[:, 1], args.eps, 1 - args.eps)
    Xw = np.concatenate([Zp, Z[train]])
    yw = np.concatenate([np.ones(len(Zp)), np.zeros(train.sum())]).astype(int)
    p_oof, w_model = crossfit(Xw, yw, args.eps)
    p = np.clip(w_model.predict_proba(Z)[:, 1], args.eps, 1 - args.eps)
    p[train] = p_oof[len(Zp):]
    w = p / (1 - p) * train.sum() / len(Zp)
    json.dump({i: [round(float(a), 5), round(float(b), 5)] for i, a, b in zip(ids, e, w)},
              open(cache / "propensity.json", "w"))
    wt = np.minimum(w[train], 10.0)
    report = {"n_pool_codes": int(len(pop)), "n_known": int(known.sum()), "verified_rate": float(yv[known].mean()),
              "e_auroc_oof": float(roc_auc_score(yv[known], e_oof)), "e_calibration": calibration(e_oof, yv[known]),
              "w_classifier_auroc_oof": float(roc_auc_score(yw, p_oof)), "e_mean_train": float(e[train].mean()),
              "w_clip_frac_train": float((w[train] > 10).mean()),
              "w_ess_frac_train": float(wt.sum() ** 2 / (wt ** 2).sum() / max(len(wt), 1)),
              "importance": dict(zip(COVARIATES, e_model.booster_.feature_importance("gain").round(1).tolist()))}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=1)
    print({k: v for k, v in report.items() if not isinstance(v, (list, dict))})


if __name__ == "__main__":
    main()
