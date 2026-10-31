import json
from pathlib import Path

import numpy as np


class Results:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def json(self, *parts) -> dict | None:
        p = self.root.joinpath(*parts)
        return json.load(open(p)) if p.exists() else None

    def summary(self, run: str) -> dict | None:
        return self.json("eval", run, "summary.json")

    def metric(self, run: str, key: str) -> tuple[float, float, int] | None:
        s = self.summary(run)
        if s is None or key not in s["metrics"]:
            return None
        m = s["metrics"][key]
        return m["mean"], m["std"], m["n"]

    def n_seeds(self, run: str) -> int:
        s = self.summary(run)
        return len(s["seeds"]) if s else 0

    def seed_results(self, run: str) -> dict:
        d = self.root / "eval" / run
        return {int(f.stem[1:]): json.load(open(f)) for f in sorted(d.glob("s*.json"))} if d.exists() else {}

    def preds(self, run: str) -> dict:
        d = self.root / "eval" / run
        return {int(f.stem.split("_s")[1]): np.load(f, allow_pickle=True) for f in sorted(d.glob("pred_s*.npz"))} \
            if d.exists() else {}

    def p_holm(self, ref: str, protocol: str, run: str, subset: str = "all") -> float | None:
        r = self.json("analysis", f"paired_{ref}_{protocol}_{subset}.json")
        return r["runs"][run]["p_holm"] if r and run in r["runs"] else None

    def run_dirs(self, run: str) -> list[Path]:
        return sorted((self.root / "runs").glob(f"{run}_s*")) if (self.root / "runs").exists() else []
