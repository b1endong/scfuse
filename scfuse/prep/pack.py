import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scfuse.labels import ANALYZERS, load_label_map, mythril_label, slither_label
from scfuse.prep.common import Bytecodes, Raw, Records, load_index, parser, pool_map
from scfuse.schema import ARRAYS, CHARS_SCALE, CHUNK, MAX_CHUNKS, Q_LAYOUT, UNITS_SCALE
from scfuse.utils.ast import LABELS
from scfuse.utils.evm import build_cfg
from scfuse.utils.solidity import scrub, squash, squash_map

TOKENIZER = "microsoft/unixcoder-base"
_G = {}


def _init(raw_root: str, cache: str, tokenizer: str):
    from transformers import AutoTokenizer
    cache = Path(cache)
    _G.update(tok=AutoTokenizer.from_pretrained(tokenizer), bc=Bytecodes(Raw(raw_root)),
              S=Records(cache / "units_S.jsonl"), D=Records(cache / "units_D.jsonl"), cfg=Records(cache / "cfg.jsonl"))


def text_arrays(token_lists: list, sels: list) -> dict:
    toks, chunks, units = [], [], []
    for ids, sel in zip(token_lists, sels):
        cs = [ids[i:i + CHUNK] for i in range(0, max(len(ids), 1), CHUNK)][:MAX_CHUNKS]
        units.append([len(chunks), len(cs), -1 if sel is None else sel])
        for c in cs:
            chunks.append([len(toks), len(c)])
            toks += c
    return {"tok": np.asarray(toks, np.uint16), "chunk": np.asarray(chunks, np.int64).reshape(-1, 2),
            "unit": np.asarray(units, np.int64).reshape(-1, 3)}


def source_tokens(units: list, gold: bool):
    tok, names = _G["tok"], {}
    if gold:
        texts = [scrub(squash(u["text"]), names) for u in units]
        return tok(texts, add_special_tokens=False)["input_ids"] if texts else [], None
    maps = [squash_map(u["text"]) for u in units]
    if not maps:
        return [], []
    enc = tok([m[0] for m in maps], add_special_tokens=False, return_offsets_mapping=True)
    bounds = []
    for (text, idx), offs in zip(maps, enc["offset_mapping"]):
        lo = np.asarray([idx[a] if a < len(idx) else len(idx) for a, _ in offs], np.int64)
        hi = np.asarray([idx[b - 1] + 1 if 0 < b <= len(idx) else lo[k] for k, (_, b) in enumerate(offs)], np.int64)
        bounds.append((lo, np.maximum.accumulate(hi) if len(hi) else hi))
    return enc["input_ids"], bounds


def span_arrays(spans: list, bounds: list, n_tokens: list, pc_block: dict) -> tuple[np.ndarray, np.ndarray]:
    span_id, pairs = {}, set()
    for pc, u, lo, hi in spans:
        b = pc_block.get(pc)
        if b is None or u >= len(bounds):
            continue
        starts, ends = bounds[u]
        cap = min(n_tokens[u], CHUNK * MAX_CHUNKS)
        t0 = int(np.searchsorted(ends, lo, side="right"))
        t1 = min(int(np.searchsorted(starts, hi, side="left")), cap)
        if t0 >= t1:
            continue
        key = (u, t0, t1)
        pairs.add((b, span_id.setdefault(key, len(span_id))))
    spans_out = np.asarray(sorted(span_id, key=span_id.get), np.int32).reshape(-1, 3)
    return np.asarray(sorted(pairs), np.int32).reshape(-1, 2), spans_out


def process(job: tuple) -> dict:
    row, labels, prop = job
    lm, cid, gold = load_label_map(), job[0]["id"], job[0]["kind"] != "weak"
    rs, rd, rg = _G["S"].get(cid) or {}, _G["D"].get(cid) or {}, _G["cfg"].get(cid) or {}
    arr = {}
    s_units = rs.get("units", [])
    s_ids, bounds = source_tokens(s_units, gold)
    for k, a in text_arrays(s_ids, [u["sel"] for u in s_units]).items():
        arr[f"S_{k}"] = a
    d_units = rd.get("units", [])
    d_ids = _G["tok"]([u["text"] for u in d_units], add_special_tokens=False)["input_ids"] if d_units else []
    for k, a in text_arrays(d_ids, [u["sel"] for u in d_units]).items():
        arr[f"D_{k}"] = a
    code = _G["bc"].get(cid)
    g = build_cfg(code) if code else None
    n = len(g.pcs) if g else 0
    arr["G_x"] = g.x if n else np.zeros((0, ARRAYS["G_x"][1]), np.uint8)
    arr["G_edge"] = g.edges if n else np.zeros((0, 2), np.int16)
    arr["G_etype"] = g.etypes if n else np.zeros(0, np.uint8)
    arr["G_fnsel"] = np.asarray(g.sels if n else [], np.int64)
    arr["G_mem"] = np.asarray([[b, f] for f, bl in enumerate(g.fn_blocks) for b in bl] if n else [],
                              np.int32).reshape(-1, 2)
    arr["G_pc"] = np.asarray(g.pcs if n else [], np.int32)
    compiled = bool(rs.get("compiled"))
    nodes, a_units = [], []
    if compiled:
        for u in s_units:
            a_units.append([len(nodes), len(u["ast"]), -1 if u["sel"] is None else u["sel"]])
            nodes += u["ast"]
    arr["A_node"] = np.asarray(nodes, np.int16).reshape(-1, 3)
    arr["A_unit"] = np.asarray(a_units, np.int64).reshape(-1, 3)
    if bounds and rs.get("spans") and n:
        arr["SP_pair"], arr["SP_span"] = span_arrays(rs["spans"], bounds, [len(x) for x in s_ids],
                                                     {pc: i for i, pc in enumerate(g.pcs)})
    else:
        arr["SP_pair"], arr["SP_span"] = np.zeros((0, 2), np.int32), np.zeros((0, 3), np.int32)
    solc = (row.get("solc") or "0.0.0").split(".")
    opt = rs.get("optimizer", -1)
    s_chars = sum(len(u["text"]) for u in s_units)
    meta = {"id": cid, "kind": row["kind"], "split": row["split"], "date": row["date"], "cluster": row["cluster"],
            "m_S": int(len(s_units) > 0), "m_G": int(n > 0), "m_D": int(len(d_units) > 0),
            "m_A": int(compiled and len(a_units) > 0),
            "y_gold": row["y_gold"] or [0] * lm.K, "audited_all": int(row.get("audited_all", 0)),
            "q_S": [int(solc[1]) / 10 if len(solc) > 1 else 0.0, int(solc[2]) / 40 if len(solc) > 2 else 0.0,
                    0.5 if opt < 0 else float(opt), float(rs.get("matched", False)),
                    float(np.log1p(len(s_units)) / UNITS_SCALE["S"]), float(np.log1p(s_chars) / CHARS_SCALE)],
            "q_G": rg.get("q_G", [0.0] * 4),
            "q_D": rd.get("q_D", [0.0] * 5),
            "q_A": [float(np.log1p(len(nodes)) / 10), float(np.log1p(len(a_units)) / UNITS_SCALE["A"]),
                    float(np.mean([u.get("ast_trunc", False) for u in s_units])) if compiled and s_units else 0.0,
                    float(rs.get("matched", False))],
            "compiled": int(compiled), "matched": int(bool(rs.get("matched"))), "lifter": rd.get("lifter", "none"),
            "D_fidelity": rd.get("fidelity", 0.0), "D_precision": rd.get("precision", 0.0),
            "D_coverage": rd.get("coverage", 0.0), "D_chars": rd.get("n_chars", 0),
            "e": prop[0] if prop else float("nan"), "w": prop[1] if prop else float("nan")}
    for a in ANALYZERS:
        y, obs = np.zeros(lm.K, np.int8), np.zeros(lm.K, np.int8)
        rec = labels.get(a)
        if rec is not None and rec["status"] == "ok":
            y = slither_label(rec["found"], lm) if a == "slither" else mythril_label(rec["found"], lm)
            obs = lm.covered[a].astype(np.int8)
        meta[f"y_{a}"], meta[f"obs_{a}"] = y.tolist(), obs.tolist()
    return {"meta": meta, "arr": arr}


def analyzer_labels(cache: Path) -> dict:
    out = {a: {} for a in ANALYZERS}
    for path in sorted((cache / "analyzers").glob("*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r["analyzer"] in out and (r["id"] not in out[r["analyzer"]] or r["status"] == "ok"):
                out[r["analyzer"]][r["id"]] = r
    return out


def main():
    ap = parser("Pack views, anchors, labels and descriptors into flat arrays")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    args = ap.parse_args()
    cache = Path(args.cache)
    out_dir = cache / "packed"
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded = json.load(open(cache / "excluded.json")) if (cache / "excluded.json").exists() else {}
    rows = [r for r in load_index(cache) if r["id"] not in excluded]
    labels = analyzer_labels(cache)
    prop = json.load(open(cache / "propensity.json")) if (cache / "propensity.json").exists() else {}
    jobs = ((r, {a: labels[a].get(r["id"]) for a in ANALYZERS}, prop.get(r["id"])) for r in rows)
    handles = {k: open(out_dir / f"{k}.bin", "wb") for k in ARRAYS}
    counts, metas, lifters = dict.fromkeys(ARRAYS, 0), [], Counter()
    for res in pool_map(process, jobs, args.workers, "pack", 8, _init, (args.raw, args.cache, args.tokenizer)):
        m = res["meta"]
        for k, a in res["arr"].items():
            m[f"{k}_0"], m[f"{k}_n"] = counts[k], len(a)
            handles[k].write(np.ascontiguousarray(a, dtype=ARRAYS[k][0]).tobytes())
            counts[k] += len(a)
        lifters[m["lifter"]] += 1
        metas.append(m)
    for h in handles.values():
        h.close()
    pd.DataFrame(metas).to_parquet(out_dir / "meta.parquet")
    json.dump({k: {"dtype": ARRAYS[k][0], "width": ARRAYS[k][1], "rows": counts[k]} for k in ARRAYS},
              open(out_dir / "arrays.json", "w"), indent=1)
    json.dump({"version": 3, "q": Q_LAYOUT, "ast_vocab": len(LABELS), "tokenizer": args.tokenizer,
               "lifters": dict(lifters)}, open(out_dir / "layout.json", "w"), indent=1)
    print(f"packed {len(metas)} contracts; rows: {counts}; lifters: {dict(lifters)}")


if __name__ == "__main__":
    main()
