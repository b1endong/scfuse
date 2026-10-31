import argparse
import json
from pathlib import Path

import numpy as np

from scfuse.config import load_config, run_name


def main():
    ap = argparse.ArgumentParser(description="Gate G2: spread of the best validation macro-F1 over the seeds of a run")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-std", type=float, default=0.05)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = Path(cfg["run"]["out"])
    rows = {}
    for s in cfg["seeds"]:
        f = out / run_name(cfg, s) / "val_best.json"
        if f.exists():
            r = json.load(open(f))
            rows[s] = {"epoch": r["epoch"], "macro_f1": r["macro_f1"], "f1": r["f1"]}
    if not rows:
        raise SystemExit(f"no val_best.json for {cfg['run']['name']} under {out}")
    f1 = np.array([r["macro_f1"] for r in rows.values()])
    res = {"run": cfg["run"]["name"], "seeds": rows, "mean": float(f1.mean()),
           "std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0, "missing": [s for s in cfg["seeds"] if s not in rows]}
    res["gate_G2"] = bool(len(f1) > 1 and res["std"] <= args.max_std)
    json.dump(res, open(out / f"{cfg['run']['name']}_seeds.json", "w"), indent=1)
    for s, r in rows.items():
        print(f"seed {s}: epoch {r['epoch']}, val macro-F1 {r['macro_f1']:.4f}")
    print(f"mean {res['mean']:.4f} std {res['std']:.4f} -> G2 {'pass' if res['gate_G2'] else 'fail'}")


if __name__ == "__main__":
    main()
