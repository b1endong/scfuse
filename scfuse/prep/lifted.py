import re
from pathlib import Path

import numpy as np

from scfuse.prep.common import Bytecodes, Raw, Records, load_index, parser, pool_map, write_jsonl
from scfuse.schema import CHARS_SCALE, MAX_UNITS, UNITS_SCALE
from scfuse.utils.evm import fidelity, selector_agreement
from scfuse.utils.solc import compile_lifted
from scfuse.utils.solidity import blank_comments, iter_units, mask_strings, squash

_SEL_NAME = re.compile(r"\b(?:func|Unresolved|unresolved)_(?:0x)?([0-9a-fA-F]{8})\b")
_SEL_NOTE = re.compile(r"[Ss]elector\b[:\s]+0x([0-9a-fA-F]{8})")
_HEX_NAME = re.compile(r"\b(func|Unresolved|unresolved|Event|CustomError)_(?:0x)?([0-9a-fA-F]{8})\b")
_PREFIX = {"func": "f", "Unresolved": "f", "unresolved": "v", "Event": "event", "CustomError": "error"}
_G = {}


def _init(raw_root: str, cache: str, lifter: str):
    raw = Raw(raw_root)
    _G.update(raw=raw, bc=Bytecodes(raw), cfg=Records(Path(cache) / "cfg.jsonl"), cache=Path(cache), lifter=lifter)


def lifted_text(cid: str) -> tuple[str | None, str]:
    qwen = _G["cache"] / "lifted" / f"{cid}.sol"
    if _G["lifter"] in ("qwen", "auto") and qwen.exists():
        return qwen.read_text(errors="replace"), "qwen"
    heim = _G["raw"].heimdall / f"{cid}.txt"
    if _G["lifter"] in ("heimdall", "auto") and heim.exists():
        return heim.read_text(errors="replace"), "heimdall"
    return None, "none"


def neutralize(text: str) -> str:
    names, counts = {}, dict.fromkeys(_PREFIX.values(), 0)

    def sub(m):
        if m.group(0) not in names:
            p = _PREFIX[m.group(1)]
            names[m.group(0)] = f"{p}{counts[p]}"
            counts[p] += 1
        return names[m.group(0)]

    return _HEX_NAME.sub(sub, text)


def units(text: str) -> list[dict]:
    blank = blank_comments(text)
    masked = mask_strings(blank)
    out, prev = [], 0
    for kw, kpos, head, close in iter_units(masked, 0, len(masked)):
        m = _SEL_NAME.search(head)
        sel = int(m.group(1), 16) if m else None
        if sel is None:
            notes = list(_SEL_NOTE.finditer(text, prev, kpos))
            sel = int(notes[-1].group(1), 16) if notes else None
        out.append({"kind": kw, "sel": sel if kw == "function" else None,
                    "text": squash(neutralize(blank[kpos:close + 1]))})
        prev = close + 1
    return out[:MAX_UNITS]


def process(row: dict) -> dict:
    cid = row["id"]
    text, lifter = lifted_text(cid)
    cfg = _G["cfg"].get(cid) or {}
    g_sels = cfg.get("sels", [])
    rec = {"id": cid, "lifter": lifter, "units": [], "n_chars": 0, "compiles": 0, "fidelity": 0.0,
           "sel_agreement": 0.0, "precision": 0.0, "coverage": 0.0}
    if not text:
        rec["q_D"] = [0.0] * 5
        return rec
    us = units(text)
    chain = _G["bc"].get(cid)
    compiled = compile_lifted(text, row.get("solc")) if chain else None
    fd = fidelity(compiled, chain) if compiled else 0.0
    agree, prec, cov = selector_agreement([u["sel"] for u in us], g_sels)
    rec.update(units=us, n_chars=len(text), compiles=int(compiled is not None), fidelity=round(fd, 5),
               sel_agreement=round(agree, 5), precision=round(prec, 5), coverage=round(cov, 5))
    rec["q_D"] = [float(rec["compiles"]), rec["fidelity"], rec["sel_agreement"],
                  float(np.log1p(len(us)) / UNITS_SCALE["D"]), float(np.log1p(len(text)) / CHARS_SCALE)]
    return rec


def main():
    ap = parser("View D: lifted function units, round-trip fidelity and selector agreement")
    ap.add_argument("--lifter", choices=("auto", "qwen", "heimdall"), default="auto")
    args = ap.parse_args()
    rows = load_index(args.cache)
    n = write_jsonl(Path(args.cache) / "units_D.jsonl",
                    pool_map(process, rows, args.workers, "lifted", 8, _init, (args.raw, args.cache, args.lifter)))
    print(f"lifted: {n} contracts")


if __name__ == "__main__":
    main()
