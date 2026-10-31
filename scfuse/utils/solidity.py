import re

from eth_utils import keccak

_LEX = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?(?:\*/|\Z)', re.S)
_NOT_NL = re.compile(r"[^\n]")
_DECL = re.compile(r"\b(abstract\s+contract|contract|library|interface)\s+([A-Za-z_$][\w$]*)")
_UNIT = re.compile(r"\b(function|modifier|constructor|fallback|receive)\b")
_TYPE = re.compile(r"^\s*([A-Za-z_$][\w$.]*)\s*((?:\[[^\]]*\]\s*)*)")
_IDENT = re.compile(r"\b[A-Za-z_$][\w$]*\b")
_ELEM = re.compile(r"^(address|bool|string|bytes([1-9]|[12]\d|3[0-2])?|u?int(8|16|24|32|40|48|56|64|72|80|88|96|104|"
                   r"112|120|128|136|144|152|160|168|176|184|192|200|208|216|224|232|240|248|256)?)$")
_DROP = re.compile(r"\b(memory|storage|calldata|indexed|payable)\b")
_VIS = ("external", "public", "internal", "private")
_PAIR = {"(": ")", "{": "}", "[": "]"}

LEAKY_NAME = re.compile(
    r"re_?ent|entrancy|intou|vundflw|tmstmp|unchk|tx_?origin|bugv?_|overflow|underflow|vulnerab|insecure|suicid"
    r"|^dos|dos$|_dos|orderdependence|unprotected|phishable|missing|attack|(?-i:_TOD|TOD_|^TOD)", re.I)


def blank_comments(src: str) -> str:
    return _LEX.sub(lambda m: m.group() if m.group()[0] in "\"'" else _NOT_NL.sub(" ", m.group()), src)


def mask_strings(src: str) -> str:
    def sub(m):
        s = m.group()
        return s[0] + "x" * (len(s) - 2) + s[-1] if s[0] in "\"'" else _NOT_NL.sub(" ", s)
    return _LEX.sub(sub, src)


def match_close(s: str, i: int) -> int:
    o, c, depth = s[i], _PAIR[s[i]], 0
    for m in re.finditer(re.escape(o) + "|" + re.escape(c), s[i:]):
        depth += 1 if m.group() == o else -1
        if depth == 0:
            return i + m.start()
    return len(s) - 1


def split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return out


def squash(text: str) -> str:
    return " ".join(text.split())


def squash_map(text: str) -> tuple[str, list[int]]:
    out, idx = [], []
    for m in re.finditer(r"\S+", text):
        if out:
            out.append(" ")
            idx.append(m.start() - 1)
        out.append(m.group())
        idx.extend(range(m.start(), m.end()))
    return "".join(out), idx


def scrub(text: str, names: dict) -> str:
    return _IDENT.sub(lambda m: names.setdefault(m.group(), f"v{len(names)}") if LEAKY_NAME.search(m.group())
                      else m.group(), text)


def selector(name: str, types: list[str]) -> int:
    return int.from_bytes(keccak(text=f"{name}({','.join(types)})")[:4], "big")


def canonical_type(param: str, idx: dict) -> str | None:
    m = _TYPE.match(_DROP.sub(" ", param))
    if not m:
        return None
    base, arr = m.group(1).split(".")[-1], re.sub(r"\s", "", m.group(2))
    base = {"uint": "uint256", "int": "int256", "byte": "bytes1"}.get(base, base)
    if base in idx["enums"]:
        base = "uint8"
    elif base in idx["contracts"]:
        base = "address"
    elif base in idx["structs"] or not _ELEM.match(base):
        return None
    return base + arr


def iter_units(masked: str, start: int, end: int):
    pos = start
    while (m := _UNIT.search(masked, pos, end)):
        stop = re.compile(r"[{;]").search(masked, m.end(), end)
        if stop is None:
            return
        head = masked[m.start():stop.start()]
        while head.count("(") > head.count(")") and stop is not None:
            stop = re.compile(r"[{;]").search(masked, stop.end(), end)
            head = masked[m.start():stop.start()] if stop else head
        if stop is None:
            return
        if masked[stop.start()] == ";":
            pos = stop.end()
            continue
        close = match_close(masked, stop.start())
        yield m.group(1), m.start(), head, close
        pos = close + 1


def scan(src: str) -> dict:
    blank = blank_comments(src)
    masked = mask_strings(blank)
    decls, pos = [], 0
    while (m := _DECL.search(masked, pos)):
        brace = masked.find("{", m.end())
        if brace < 0:
            break
        parts = re.split(r"\bis\b", masked[m.end():brace], 1)
        bases = [b.split("(")[0].strip() for b in split_top(parts[1])] if len(parts) == 2 else []
        close = match_close(masked, brace)
        decls.append(("abstract" if "abstract" in m.group(1) else m.group(1), m.group(2), bases, brace, close))
        pos = close + 1
    idx = {"structs": set(re.findall(r"\bstruct\s+([A-Za-z_$][\w$]*)", masked)),
           "enums": set(re.findall(r"\benum\s+([A-Za-z_$][\w$]*)", masked)), "contracts": {d[1] for d in decls}}
    contracts, units = [], []
    for ci, (kind, cname, bases, brace, close) in enumerate(decls):
        contracts.append({"name": cname, "kind": kind, "bases": bases})
        if kind == "interface":
            continue
        for kw, kpos, head, bclose in iter_units(masked, brace + 1, close):
            nm = re.match(r"\s*([A-Za-z_$][\w$]*)?\s*\(", head[len(kw):])
            name = nm.group(1) if nm and nm.group(1) else ""
            p_open = head.find("(")
            p_close = p_open + match_close(head[p_open:], 0) if p_open >= 0 else -1
            tail = head[p_close + 1:] if p_close >= 0 else ""
            vis = next((v for v in _VIS if re.search(rf"\b{v}\b", tail)), "public")
            ukind = kw if kw != "function" or (name and name != cname) else ("fallback" if not name else "constructor")
            sel = None
            if ukind == "function" and vis in ("public", "external") and kind != "library":
                types = [canonical_type(p, idx) for p in split_top(head[p_open + 1:p_close])]
                sel = selector(name, types) if all(t is not None for t in types) else None
            units.append({"c": ci, "kind": ukind, "name": name, "vis": vis, "sel": sel, "start": kpos,
                          "end": bclose + 1})
    return {"contracts": contracts, "units": units, "blank": blank}


def linearize(ci: int, contracts: list[dict]) -> list[int]:
    by_name = {c["name"]: i for i, c in enumerate(contracts)}
    order, queue = [], [ci]
    while queue:
        i = queue.pop(0)
        if i not in order:
            order.append(i)
            queue += [by_name[b] for b in contracts[i]["bases"] if b in by_name]
    return order


def order_scanned(rec: dict, g_sels: set, stem: str) -> list[tuple[int, int | None]]:
    contracts, units = rec["contracts"], rec["units"]
    cands = [i for i, c in enumerate(contracts) if c["kind"] in ("contract", "abstract")]
    if not cands:
        return [(j, None) for j in range(len(units))]

    def score(i):
        lin = set(linearize(i, contracts))
        return (len({u["sel"] for u in units if u["c"] in lin and u["sel"] is not None} & g_sels),
                contracts[i]["name"] == stem, i)

    lin = linearize(max(cands, key=score), contracts)
    rank = {c: r for r, c in enumerate(lin)}
    seen, out = set(), []
    for j in sorted(range(len(units)), key=lambda j: rank.get(units[j]["c"], len(lin))):
        u = units[j]
        sel = u["sel"] if u["c"] in rank and u["sel"] is not None and u["sel"] not in seen else None
        seen.add(sel)
        out.append((j, sel))
    return out
