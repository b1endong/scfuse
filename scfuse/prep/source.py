from collections import Counter
from pathlib import Path

from scfuse.prep.common import Bytecodes, Raw, load_index, parser, pool_map, write_jsonl
from scfuse.schema import MAX_UNITS
from scfuse.utils import ast as A
from scfuse.utils.evm import build_cfg, instructions, runtime, split_blocks
from scfuse.utils.solc import deployed, parse_srcmap, same_runtime, standard_json, vtuple
from scfuse.utils.solidity import blank_comments, order_scanned, scan

FILE = "main.sol"
_G = {}


def _init(raw_root: str):
    raw = Raw(raw_root)
    _G.update(raw=raw, bc=Bytecodes(raw))


def optimizer_order(settings: str) -> list[bool]:
    s = (settings or "").lower()
    if s.startswith("yes") or "true" in s:
        return [True, False]
    return [False, True]


def pick_main(contracts: dict, chain: str | None, g_sels: set, stem: str):
    best, best_key = None, None
    for name, c in contracts.items():
        obj, _, imm = deployed(c)
        if not obj:
            continue
        matched = bool(chain) and same_runtime(obj, chain, imm)
        cover = len({int(v, 16) for v in c.get("evm", {}).get("methodIdentifiers", {}).values()} & g_sels)
        key = (matched, cover, name == stem, len(obj))
        if best_key is None or key > best_key:
            best, best_key = name, key
    return best, bool(best_key and best_key[0])


def ast_units(ast: dict, legacy: bool, main_name: str, method_ids: dict) -> list[dict]:
    defs = [n for n in A.walk(ast, legacy) if A.node_type(n, legacy) == "ContractDefinition"]
    by_id = {n.get("id"): n for n in defs}
    main = next((n for n in defs if A.attrs(n, legacy).get("name") == main_name), None)
    lin = A.attrs(main, legacy).get("linearizedBaseContracts", []) if main else []
    in_lin = set(lin)
    order = [by_id[i] for i in lin if i in by_id] + [n for n in defs if n.get("id") not in in_lin]
    by_name = {}
    for sig, sel in method_ids.items():
        by_name.setdefault(sig.split("(")[0], []).append((sig, int(sel, 16)))
    seen, units = set(), []
    for c in order:
        ca = A.attrs(c, legacy)
        if ca.get("contractKind") == "interface":
            continue
        for n in A.children(c, legacy):
            t = A.node_type(n, legacy)
            if t not in ("FunctionDefinition", "ModifierDefinition"):
                continue
            a = A.attrs(n, legacy)
            if a.get("implemented") is False or (not legacy and n.get("body") is None):
                continue
            name = a.get("name") or ""
            kind = "modifier" if t == "ModifierDefinition" else a.get("kind") or (
                "constructor" if a.get("isConstructor") or name == ca.get("name") else "fallback" if not name
                else "function")
            sel = None
            if kind == "function" and c.get("id") in in_lin and a.get("visibility") in ("public", "external"):
                if a.get("functionSelector"):
                    sel = int(a["functionSelector"], 16)
                else:
                    params = next((p for p in A.children(n, legacy) if A.node_type(p, legacy) == "ParameterList"),
                                  None)
                    n_params = len(A.children(params, legacy)) if params else 0
                    cands = [s for sig, s in by_name.get(name, [])
                             if (sig.count(",") + 1 if sig[len(name) + 1:-1] else 0) == n_params]
                    sel = cands[0] if len(cands) == 1 else None
            if sel is not None and sel in seen:
                sel = None
            seen.add(sel)
            start, length, _ = A.parse_src(n.get("src"))
            seq, trunc = A.sequence(n, legacy)
            units.append({"kind": kind, "name": name, "sel": sel, "start": start, "end": start + length,
                          "ast": seq, "ast_trunc": trunc})
    return units[:MAX_UNITS]


def block_spans(compiled: str, srcmap: str, file_id: int, units: list[dict]) -> list[list[int]]:
    code = runtime(compiled)
    ins = instructions(code)
    entries = parse_srcmap(srcmap)
    span_at = {pc: e for (pc, _, _, _), e in zip(ins, entries)}
    out = []
    for bl in split_blocks(ins):
        hits = Counter()
        hull = {}
        for pc, _, _, _ in bl:
            s, n, f = span_at.get(pc, (-1, -1, -1))
            if f != file_id or s < 0 or n <= 0:
                continue
            for u, unit in enumerate(units):
                if unit["start"] <= s and s + n <= unit["end"] and not (s <= unit["start"] and s + n >= unit["end"]):
                    hits[u] += 1
                    lo, hi = hull.get(u, (s, s + n))
                    hull[u] = (min(lo, s), max(hi, s + n))
                    break
        if hits:
            u = hits.most_common(1)[0][0]
            lo, hi = hull[u]
            out.append([bl[0][0], u, lo - units[u]["start"], hi - units[u]["start"]])
    return out


def compile_units(text: str, row: dict, chain: str | None, g_sels: set, stem: str) -> dict | None:
    ver = row.get("solc")
    if not ver or vtuple(ver) < (0, 4, 11):
        return None
    try:
        utf = text.encode("latin-1").decode("utf-8")
    except UnicodeDecodeError:
        return None
    fallback = None
    for opt in optimizer_order(row.get("settings")):
        out = standard_json(ver, {FILE: utf}, opt)
        if out is None:
            continue
        contracts = out.get("contracts", {}).get(FILE, {})
        main, matched = pick_main(contracts, chain, g_sels, stem)
        if main is None:
            continue
        src = out.get("sources", {}).get(FILE, {})
        tree = src.get("ast") or src.get("legacyAST")
        if not tree:
            continue
        res = {"out": out, "main": main, "matched": matched, "optimizer": int(opt), "tree": tree,
               "file_id": src.get("id", 0)}
        if matched:
            return res
        fallback = fallback or res
    return fallback


def process(row: dict) -> dict:
    raw, bc = _G["raw"], _G["bc"]
    rec = {"id": row["id"], "solc": row.get("solc"), "compiled": False, "matched": False, "optimizer": -1,
           "units": [], "spans": []}
    path = raw.path(row.get("sol_path"))
    if path is None or not path.exists():
        return rec
    text = path.read_bytes().decode("latin-1")
    blank = blank_comments(text)
    chain = bc.get(row["id"])
    g_sels = set(build_cfg(chain).sels) if chain else set()
    stem = path.stem.split("_", 1)[-1]
    res = compile_units(text, row, chain, g_sels, stem)
    if res is not None:
        legacy = A.is_legacy(res["tree"])
        c = res["out"]["contracts"][FILE][res["main"]]
        units = ast_units(res["tree"], legacy, res["main"], c.get("evm", {}).get("methodIdentifiers", {}))
        if units:
            rec.update(compiled=True, matched=res["matched"], optimizer=res["optimizer"], main=res["main"])
            for u in units:
                u["text"] = blank[u["start"]:u["end"]]
            rec["units"] = units
            if res["matched"] and row["kind"] == "weak" and row["split"] == "train":
                obj, srcmap, _ = deployed(c)
                rec["spans"] = block_spans(obj, srcmap, res["file_id"], units)
            return rec
    sc = scan(text)
    units = []
    for j, sel in order_scanned(sc, g_sels, stem)[:MAX_UNITS]:
        u = sc["units"][j]
        units.append({"kind": u["kind"], "name": u["name"], "sel": sel, "start": u["start"], "end": u["end"],
                      "text": sc["blank"][u["start"]:u["end"]]})
    rec["units"] = units
    return rec


def main():
    ap = parser("Views S and A: solc AST function units, compiler match and block-span anchors")
    args = ap.parse_args()
    rows = load_index(args.cache)
    stats = Counter()

    def records():
        for rec in pool_map(process, rows, args.workers, "source", 4, _init, (args.raw,)):
            stats["contracts"] += 1
            stats["with_units"] += bool(rec["units"])
            stats["compiled"] += rec["compiled"]
            stats["matched"] += rec["matched"]
            stats["with_spans"] += bool(rec["spans"])
            yield rec

    write_jsonl(Path(args.cache) / "units_S.jsonl", records())
    print(dict(stats))


if __name__ == "__main__":
    main()
