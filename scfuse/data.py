import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scfuse.labels import ANALYZERS, load_label_map
from scfuse.schema import CHARS_SCALE, CHUNK, LABEL_FIXES, Q_LAYOUT_V25, UNITS_SCALE, VI, VIEWS


@dataclass
class Packed:
    meta: pd.DataFrame
    arr: dict
    layout: dict = field(default_factory=dict)


def fix_labels(meta: pd.DataFrame, fixes) -> pd.DataFrame:
    codes = load_label_map().codes
    y = np.stack(meta.y_gold.to_numpy()).copy()
    for name in fixes:
        prefix, old, new = LABEL_FIXES[name]
        rows = meta.id.str.startswith(prefix).to_numpy()
        y[rows, codes.index(old)], y[rows, codes.index(new)] = 0, 1
    meta = meta.copy()
    meta["y_gold"] = list(y)
    return meta


def load_packed(cache: str | Path, label_fix=()) -> Packed:
    d = Path(cache) / "packed"
    arr = {}
    for k, s in json.load(open(d / "arrays.json")).items():
        shape = (s["rows"], s["width"]) if s["width"] > 1 else (s["rows"],)
        arr[k] = (np.memmap(d / f"{k}.bin", dtype=s["dtype"], mode="r", shape=shape) if s["rows"]
                  else np.zeros(shape, s["dtype"]))
    layout = json.load(open(d / "layout.json")) if (d / "layout.json").exists() else {"version": 2.5, "q": Q_LAYOUT_V25}
    meta = pd.read_parquet(d / "meta.parquet")
    return Packed(fix_labels(meta, label_fix) if label_fix else meta, arr, layout)


def _col(m: pd.DataFrame, name: str, width: int) -> np.ndarray:
    if name not in m.columns:
        return np.zeros((len(m), width), np.float32)
    return np.stack(m[name].to_numpy()).astype(np.float32)


class ContractDataset(torch.utils.data.Dataset):
    def __init__(self, p: Packed, mode: str = "cached", feats_dir: str | Path | None = None,
                 unit_cap: int | None = None, tok_cap: int | None = None, ast_units: int = 32, ast_nodes: int = 256):
        self.p, self.mode, self.unit_cap, self.tok_cap = p, mode, unit_cap, tok_cap
        self.ast_units, self.ast_nodes = ast_units, ast_nodes
        m = p.meta
        fd = Path(feats_dir) if feats_dir else None
        self.feats = ({v: np.load(fd / f"{v}.npy", mmap_mode="r") for v in ("S", "D", "SP")
                       if fd and (fd / f"{v}.npy").exists()} if mode == "cached" else {})
        self.off = {c: m[c].to_numpy() for c in m.columns if c.endswith("_0") or c.endswith("_n")}
        self.layout = p.layout["q"]
        self.m = np.stack([m[f"m_{v}"].to_numpy() if f"m_{v}" in m.columns else np.zeros(len(m)) for v in VIEWS],
                          1).astype(np.float32)
        self.q = {v: _col(m, f"q_{v}", len(self.layout[v])) for v in VIEWS}
        self.q_dims = {v: self.q[v].shape[1] for v in VIEWS}
        self.y_gold = np.stack(m["y_gold"].to_numpy()).astype(np.float32)
        self.audited_all = (m["audited_all"].to_numpy(np.float32) if "audited_all" in m.columns
                            else np.zeros(len(m), np.float32))
        self.is_gold = m["kind"].ne("weak").to_numpy(np.float32)
        self.y_an = np.stack([np.stack(m[f"y_{a}"].to_numpy()) for a in ANALYZERS], 1).astype(np.float32)
        self.obs_an = np.stack([np.stack(m[f"obs_{a}"].to_numpy()) for a in ANALYZERS], 1).astype(np.float32)
        self.e = m["e"].to_numpy(np.float32).copy()
        self.w = m["w"].to_numpy(np.float32).copy()
        self.has_spans = "SP_pair_n" in self.off and "SP" in self.feats if mode == "cached" else "SP_pair_n" in self.off

    def __len__(self):
        return len(self.p.meta)

    def rows(self, k: str, i: int) -> np.ndarray:
        if f"{k}_0" not in self.off:
            w = self.p.arr[k].shape[1:] if k in self.p.arr else ()
            return np.zeros((0, *w), np.int64)
        a0, n = self.off[f"{k}_0"][i], self.off[f"{k}_n"][i]
        return self.p.arr[k][a0:a0 + n]

    def dispatcher(self, i: int) -> set:
        return set(self.rows("G_fnsel", i).tolist())

    def text_view(self, v: str, i: int, keep: np.ndarray | None = None) -> dict:
        units = self.rows(f"{v}_unit", i)
        if self.mode == "cached" and v not in self.feats:
            units = units[:0]
        rows = np.arange(len(units))
        if keep is not None:
            rows = keep
        elif self.mode == "tokens" and self.unit_cap and len(units) > self.unit_cap:
            rows = np.sort(torch.randperm(len(units))[:self.unit_cap].numpy())
        units = units[rows]
        out = {"sel": units[:, 2].copy(), "rows": rows}
        if self.mode == "cached":
            g = self.off[f"{v}_unit_0"][i] + rows
            out["feat"] = np.asarray(self.feats[v][g]) if v in self.feats else np.zeros((0, 768), np.float16)
        else:
            chunks, toks = self.rows(f"{v}_chunk", i), self.rows(f"{v}_tok", i)
            n_ch = np.minimum(units[:, 1], 1) if self.tok_cap else units[:, 1]
            cap = self.tok_cap or 1 << 30
            out["chunks"] = [np.asarray(toks[s:s + min(n, cap)], np.int64)
                             for c0, k in zip(units[:, 0], n_ch) for s, n in chunks[c0:c0 + k]]
            out["chunk_unit"] = np.repeat(np.arange(len(units)), n_ch)
            out["chunk_k"] = np.concatenate([np.arange(k) for k in n_ch]) if len(n_ch) else np.zeros(0, int)
        return out

    def graph(self, i: int) -> dict:
        return {"x": np.array(self.rows("G_x", i), np.int64), "edge": np.array(self.rows("G_edge", i)),
                "etype": np.array(self.rows("G_etype", i)), "fnsel": np.array(self.rows("G_fnsel", i)),
                "mem": np.array(self.rows("G_mem", i))}

    def ast(self, i: int) -> dict:
        units, nodes = self.rows("A_unit", i)[:self.ast_units], self.rows("A_node", i)
        return {"sel": units[:, 2].copy() if len(units) else np.zeros(0, np.int64),
                "nodes": [np.asarray(nodes[a:a + min(n, self.ast_nodes)], np.int64) for a, n, _ in units]}

    def spans(self, i: int, s_rows: np.ndarray) -> dict:
        pair, span = np.asarray(self.rows("SP_pair", i), np.int64), np.asarray(self.rows("SP_span", i), np.int64)
        empty = {"pair": np.zeros((0, 2), np.int64), "span": np.zeros((0, 3), np.int64)}
        if not len(pair):
            return empty
        new = np.full(int(span[:, 0].max()) + 1, -1)
        ok_rows = s_rows[s_rows < len(new)]
        new[ok_rows] = np.flatnonzero(s_rows < len(new))
        unit = new[span[:, 0]]
        lo, hi = span[:, 1], span[:, 2]
        if self.mode == "tokens":
            cap = self.tok_cap or CHUNK
            hi = np.minimum(hi, cap)
        keep = (unit >= 0) & (lo < hi)
        remap = np.cumsum(keep) - 1
        pair = pair[keep[pair[:, 1]]]
        out = {"pair": np.stack([pair[:, 0], remap[pair[:, 1]]], 1) if len(pair) else empty["pair"],
               "span": np.stack([unit[keep], lo[keep], hi[keep]], 1)}
        if self.mode == "cached":
            out["feat"] = np.asarray(self.feats["SP"][self.off["SP_span_0"][i] + np.flatnonzero(keep)])
        return out

    def __getitem__(self, i: int) -> dict:
        S = self.text_view("S", i)
        item = {"idx": i, "S": S, "D": self.text_view("D", i), "G": self.graph(i), "A": self.ast(i)}
        if self.has_spans:
            item["SP"] = self.spans(i, S["rows"])
        return item


def chars(q_d: np.ndarray, layout: tuple) -> float:
    return float(np.expm1(q_d[layout.index("log_chars")] * CHARS_SCALE)) if "log_chars" in layout else 0.0


def requalify(ds: ContractDataset, i: int, q_src: np.ndarray, sel: np.ndarray, n_units: int,
              n_chars: float) -> np.ndarray:
    lay, q = ds.layout["D"], q_src.copy()
    g, s = ds.dispatcher(i), {int(x) for x in sel.tolist() if x >= 0}
    both = len(s & g)
    values = {"sel_agreement": both / max(len(s | g), 1), "sel_recovery": both / len(g) if g else 0.0,
              "log_units": np.log1p(n_units) / UNITS_SCALE["D"], "log_chars": np.log1p(n_chars) / CHARS_SCALE}
    for name, val in values.items():
        if name in lay:
            q[lay.index(name)] = val
    return q.astype(np.float32)


def corrupt_d(ds: ContractDataset, i: int, kind: str, rng: np.random.Generator, donor: int | None = None) -> dict | None:
    """kind: lift (donor's whole lift), units (recipient's unit count drawn from the donor), half (own units halved)."""
    own = ds.text_view("D", i)
    n, lay = len(own["sel"]), ds.layout["D"]
    if kind == "half":
        if n < 2:
            return None
        keep = np.sort(rng.choice(n, int(np.ceil(n / 2)), replace=False))
        D = ds.text_view("D", i, keep)
        q = requalify(ds, i, ds.q["D"][i], D["sel"], len(keep), chars(ds.q["D"][i], lay) * len(keep) / n)
        return {"D": D, "qD": q}
    src = ds.text_view("D", donor)
    m = len(src["sel"])
    if m == 0:
        return None
    if kind == "lift":
        q = requalify(ds, i, ds.q["D"][donor], src["sel"], m, chars(ds.q["D"][donor], lay))
        return {"D": src, "qD": q}
    if n == 0:
        return None
    pick = rng.integers(m, size=n)
    D = ds.text_view("D", donor, src["rows"][pick])
    q = requalify(ds, i, ds.q["D"][donor], D["sel"], n, chars(ds.q["D"][donor], lay) * n / m)
    return {"D": D, "qD": q}


class CorruptTrain(torch.utils.data.Dataset):
    def __init__(self, ds: ContractDataset, rows, p: float, seed: int):
        self.ds, self.p, self.seed, self.rng = ds, p, seed, None
        self.donors = np.asarray([r for r in rows if len(ds.rows("D_unit", int(r)))], int)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        if self.rng is None:
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng([self.seed, info.id if info else 0])
        item, rng = self.ds[i], self.rng
        item["corrupt_target"] = 0.0
        if not self.ds.m[i, VI["D"]] or rng.random() >= self.p:
            return item
        if rng.random() < 0.5:
            pool = self.donors[self.donors != i]
            res = corrupt_d(self.ds, i, "units", rng, int(rng.choice(pool))) if len(pool) else None
        else:
            res = corrupt_d(self.ds, i, "half", rng)
        if res is not None:
            item["D"], item["q_override"], item["corrupt_target"] = res["D"], {"D": res["qD"]}, 1.0
        return item


def _pad_text(items, v, mode):
    B, U = len(items), max(1, max(len(it[v]["sel"]) for it in items))
    out = {"sel": torch.full((B, U), -1, dtype=torch.long), "mask": torch.zeros(B, U, dtype=torch.bool)}
    for b, it in enumerate(items):
        n = len(it[v]["sel"])
        out["sel"][b, :n] = torch.from_numpy(np.asarray(it[v]["sel"], np.int64))
        out["mask"][b, :n] = True
    if mode == "cached":
        H = next((it[v]["feat"].shape[1] for it in items if len(it[v]["feat"])), 768)
        out["feat"] = torch.zeros(B, U, H)
        for b, it in enumerate(items):
            out["feat"][b, :len(it[v]["feat"])] = torch.from_numpy(np.asarray(it[v]["feat"], np.float32))
        return out
    chunks = [(b, u, k, c) for b, it in enumerate(items)
              for c, u, k in zip(it[v]["chunks"], it[v]["chunk_unit"], it[v]["chunk_k"])]
    L = max([len(c) for *_, c in chunks] + [1])
    out["tok"] = torch.zeros(len(chunks), L, dtype=torch.long)
    out["tok_len"] = torch.tensor([len(c) for *_, c in chunks], dtype=torch.long)
    out["chunk_unit"] = torch.tensor([b * U + u for b, u, _, _ in chunks], dtype=torch.long)
    out["chunk_of"] = {(b, u, k): r for r, (b, u, k, _) in enumerate(chunks)}
    for r, (*_, c) in enumerate(chunks):
        out["tok"][r, :len(c)] = torch.from_numpy(c)
    return out


def _pad_graph(items):
    B = len(items)
    N = max(1, max(len(it["G"]["x"]) for it in items))
    F = max(len(it["G"]["fnsel"]) for it in items) + 1
    g = {"x": torch.zeros(B, N, 32, dtype=torch.long), "mask": torch.zeros(B, N, dtype=torch.bool),
         "adj": torch.zeros(B, N, N, dtype=torch.uint8), "fn": torch.zeros(B, F, N, dtype=torch.bool),
         "fnsel": torch.full((B, F), -1, dtype=torch.long), "fnmask": torch.zeros(B, F, dtype=torch.bool)}
    for b, it in enumerate(items):
        G, n, f = it["G"], len(it["G"]["x"]), len(it["G"]["fnsel"])
        if n == 0:
            continue
        g["x"][b, :n] = torch.from_numpy(G["x"])
        g["mask"][b, :n] = True
        e = torch.from_numpy(G["edge"].astype(np.int64)).reshape(-1, 2)
        if len(e):
            g["adj"][b, e[:, 0], e[:, 1]] = torch.from_numpy(G["etype"].astype(np.uint8)) + 1
        mem = torch.from_numpy(G["mem"].astype(np.int64)).reshape(-1, 2)
        if len(mem):
            g["fn"][b, mem[:, 1], mem[:, 0]] = True
        g["fnsel"][b, :f] = torch.from_numpy(G["fnsel"].astype(np.int64))
        g["fnmask"][b, :f] = True
        resid = ~g["fn"][b, :f, :n].any(0)
        g["fn"][b, F - 1, :n] = resid
        g["fnmask"][b, F - 1] = bool(resid.any())
    return g


def _pad_ast(items):
    B, U = len(items), max(1, max(len(it["A"]["sel"]) for it in items))
    rows = [(b, u, x) for b, it in enumerate(items) for u, x in enumerate(it["A"]["nodes"]) if len(x)]
    L = max([len(x) for *_, x in rows] + [1])
    out = {"sel": torch.full((B, U), -1, dtype=torch.long), "mask": torch.zeros(B, U, dtype=torch.bool),
           "node": torch.zeros(len(rows), L, 3, dtype=torch.long),
           "row": torch.tensor([b * U + u for b, u, _ in rows], dtype=torch.long)}
    for b, it in enumerate(items):
        n = len(it["A"]["sel"])
        out["sel"][b, :n] = torch.from_numpy(np.asarray(it["A"]["sel"], np.int64))
        out["mask"][b, :n] = torch.tensor([len(x) > 0 for x in it["A"]["nodes"]], dtype=torch.bool)
    for r, (_, _, x) in enumerate(rows):
        out["node"][r, :len(x)] = torch.from_numpy(x)
    return out


def _pad_spans(items, s_batch, mode, N):
    blocks, span_ref, sb, su, lo, hi, feats, base = [], [], [], [], [], [], [], 0
    for b, it in enumerate(items):
        sp = it.get("SP")
        if sp is None or not len(sp["pair"]):
            continue
        blocks.append(b * N + sp["pair"][:, 0])
        span_ref.append(base + sp["pair"][:, 1])
        sb.append(np.full(len(sp["span"]), b))
        su.append(sp["span"][:, 0])
        lo.append(sp["span"][:, 1])
        hi.append(sp["span"][:, 2])
        if mode == "cached":
            feats.append(np.asarray(sp["feat"], np.float32))
        base += len(sp["span"])
    if not base:
        return None
    cat = lambda xs: torch.from_numpy(np.concatenate(xs).astype(np.int64))
    out = {"block": cat(blocks), "span": cat(span_ref), "b": cat(sb), "unit": cat(su), "lo": cat(lo), "hi": cat(hi)}
    if mode == "cached":
        out["feat"] = torch.from_numpy(np.concatenate(feats))
        out["valid"] = torch.ones(base, dtype=torch.bool)
    else:
        k = torch.div(out["lo"], CHUNK, rounding_mode="floor")
        rows = [s_batch["chunk_of"].get((b, u, c), -1)
                for b, u, c in zip(out["b"].tolist(), out["unit"].tolist(), k.tolist())]
        out["chunk"] = torch.tensor(rows, dtype=torch.long)
        out["lo"] = out["lo"] - k * CHUNK
        out["hi"] = torch.clamp(out["hi"] - k * CHUNK, max=CHUNK)
        out["valid"] = (out["chunk"] >= 0) & (out["lo"] < out["hi"])
    return out


class Collate:
    def __init__(self, ds: ContractDataset):
        self.ds = ds

    def __call__(self, items):
        ds, idx = self.ds, np.array([it["idx"] for it in items])
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[idx]))
        q = {v: t(ds.q[v]) for v in VIEWS}
        m = t(ds.m)
        for b, it in enumerate(items):
            for v, vec in it.get("q_override", {}).items():
                q[v][b] = torch.from_numpy(vec)
            for v, val in it.get("m_override", {}).items():
                m[b, VI[v]] = val
        S = _pad_text(items, "S", ds.mode)
        G = _pad_graph(items)
        out = {"idx": torch.from_numpy(idx), "S": S, "D": _pad_text(items, "D", ds.mode), "G": G,
               "A": _pad_ast(items), "m": m, "q": q, "e": t(ds.e), "w": t(ds.w), "y_gold": t(ds.y_gold),
               "audited_all": t(ds.audited_all), "is_gold": t(ds.is_gold), "y_an": t(ds.y_an), "obs_an": t(ds.obs_an)}
        sp = _pad_spans(items, S, ds.mode, G["x"].shape[1]) if any("SP" in it for it in items) else None
        if sp is not None:
            out["SP"] = sp
        S.pop("chunk_of", None)
        if "corrupt_target" in items[0]:
            out["corrupt_target"] = torch.tensor([it["corrupt_target"] for it in items], dtype=torch.float32)
        return out


class ContractSampler(torch.utils.data.Sampler):
    def __init__(self, weak_idx, gold_idx, batch_size: int, n_gold: int, seed: int):
        self.weak, self.gold = np.asarray(weak_idx), np.asarray(gold_idx)
        self.B, self.seed, self.epoch = batch_size, seed, 0
        self.n_gold = min(n_gold, len(self.gold)) if len(self.weak) else batch_size
        self.n_weak = self.B - self.n_gold

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.weak) / self.n_weak) if len(self.weak) else math.ceil(len(self.gold) / self.B)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        weak = rng.permutation(self.weak)
        reps = len(self) * self.n_gold // max(1, len(self.gold)) + 1
        gold = np.concatenate([rng.permutation(self.gold) for _ in range(reps)]) if len(self.gold) else self.gold
        for b in range(len(self)):
            yield weak[b * self.n_weak:(b + 1) * self.n_weak].tolist() + gold[b * self.n_gold:(b + 1) * self.n_gold].tolist()


def availability(m_real: torch.Tensor, e: torch.Tensor, mode: str, gen: torch.Generator, e_bar: float = 0.5,
                 drop_d: float = 0.0) -> torch.Tensor:
    m = m_real.clone()
    s, d, a = VI["S"], VI["D"], VI["A"]
    if drop_d > 0:
        m[:, d] *= torch.bernoulli(torch.full_like(e, 1 - drop_d), generator=gen)
    if mode == "propensity":
        m[:, s] *= torch.bernoulli(e.clamp(0, 1), generator=gen)
    elif mode == "uniform":
        m[:, s] *= torch.bernoulli(torch.full_like(e, e_bar), generator=gen)
    elif mode == "bytecode":
        m[:, s] = 0
    elif mode != "full":
        raise ValueError(mode)
    m[:, a] *= m[:, s]
    return m
