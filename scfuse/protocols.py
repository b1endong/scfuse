import json
from pathlib import Path

import numpy as np
import torch

from scfuse.data import ContractDataset, chars, corrupt_d, requalify
from scfuse.schema import CHARS_SCALE, VI

MASKS = {"P1": ("full", 1), "P2": ("propensity", 10), "P3": ("uniform", 10), "P4": ("bytecode", 1)}
CORRUPTIONS = {"P5_replaced": ("lift", 5), "P5_half": ("half", 5)}
RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
P6_KINDS = ("absent", "lift", "half")


class Override(torch.utils.data.Dataset):
    def __init__(self, ds: ContractDataset, D: dict | None = None, q: dict | None = None, m: dict | None = None):
        self.ds, self.D, self.q, self.m = ds, D or {}, q or {}, m or {}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        if i in self.D:
            item["D"] = self.D[i]
        if i in self.q:
            item["q_override"] = {"D": self.q[i]}
        if i in self.m:
            item["m_override"] = {"D": self.m[i]}
        return item


def derangement(rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(len(rows))
    while len(rows) > 1 and (perm == np.arange(len(rows))).any():
        perm = rng.permutation(len(rows))
    return rows[perm]


def corrupt(ds: ContractDataset, rows: np.ndarray, kind: str, seed: int) -> Override:
    rng = np.random.default_rng(seed)
    if kind == "absent":
        return Override(ds, m={int(i): 0.0 for i in rows})
    donors = derangement(rows, rng) if kind == "lift" else rows
    D, q = {}, {}
    for i, j in zip(rows.tolist(), donors.tolist()):
        if not ds.m[i, VI["D"]]:
            continue
        res = corrupt_d(ds, i, kind, rng, j)
        if res is not None:
            D[i], q[i] = res["D"], res["qD"]
    return Override(ds, D, q)


def sel_index(ds: ContractDataset) -> int:
    lay = ds.layout["D"]
    return lay.index("sel_agreement") if "sel_agreement" in lay else lay.index("sel_recovery")


def gate_threshold(ds: ContractDataset, val_rows: np.ndarray, max_reject: float = 0.02) -> float:
    rows = val_rows[ds.m[val_rows, VI["D"]] > 0]
    f = np.sort(ds.q["D"][rows, sel_index(ds)])
    return float(f[int(np.floor(max_reject * len(f)))]) if len(f) else 0.0


def gated(ov: Override, rows: np.ndarray, tau_gate: float) -> Override:
    k = sel_index(ov.ds)
    m = dict(ov.m)
    for i in rows.tolist():
        q = ov.q.get(i, ov.ds.q["D"][i])
        if ov.ds.m[i, VI["D"]] and q[k] < tau_gate:
            m[i] = 0.0
    return Override(ov.ds, ov.D, ov.q, m)


def mutated(ds: ContractDataset, manifest: list, draw: int, feats: np.ndarray) -> tuple[Override, np.ndarray]:
    D, q, valid = {}, {}, []
    lay = ds.layout["D"]
    for r in manifest:
        if r["draw"] != draw or not r["valid"]:
            continue
        i = int(r["row"])
        view = ds.text_view("D", i)
        if r["unit"] >= len(view["sel"]):
            continue
        view["feat"] = view["feat"].copy()
        view["feat"][r["unit"]] = feats[r["feat_row"]]
        D[i], valid = view, valid + [i]
        n_chars = chars(ds.q["D"][i], lay) + r["delta_chars"]
        q[i] = requalify(ds, i, ds.q["D"][i], view["sel"], len(view["sel"]), max(n_chars, 0.0))
        if "log_chars" in lay:
            q[i][lay.index("log_chars")] = np.log1p(max(n_chars, 0.0)) / CHARS_SCALE
    return Override(ds, D, q), np.asarray(valid, int)


def load_mutations(cache: Path, feats_dir: Path):
    man, f = cache / "mutations.json", feats_dir / "D_mut.npy"
    if not man.exists() or not f.exists():
        return None, None
    return json.load(open(man)), np.load(f, mmap_mode="r")


def broken_share(ds: ContractDataset, rows: np.ndarray, kind: str, rho: float, seed: int) -> Override:
    order = np.random.default_rng(seed).permutation(rows)
    hit = np.sort(order[:int(round(rho * len(rows)))])
    if not len(hit):
        return Override(ds)
    if kind == "lift":
        rng = np.random.default_rng(seed + 1)
        D, q = {}, {}
        for i in hit.tolist():
            if not ds.m[i, VI["D"]]:
                continue
            j = int(rng.choice(rows[rows != i]))
            res = corrupt_d(ds, i, "lift", rng, j)
            if res is not None:
                D[i], q[i] = res["D"], res["qD"]
        return Override(ds, D, q)
    return corrupt(ds, hit, kind, seed + 1)
