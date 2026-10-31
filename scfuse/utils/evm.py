from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from scfuse.schema import BLOCK_OPS, MAX_BLOCKS

OPS = {
    0x00: ("STOP", 0, 0), 0x01: ("ADD", 2, 1), 0x02: ("MUL", 2, 1), 0x03: ("SUB", 2, 1), 0x04: ("DIV", 2, 1),
    0x05: ("SDIV", 2, 1), 0x06: ("MOD", 2, 1), 0x07: ("SMOD", 2, 1), 0x08: ("ADDMOD", 3, 1), 0x09: ("MULMOD", 3, 1),
    0x0A: ("EXP", 2, 1), 0x0B: ("SIGNEXTEND", 2, 1),
    0x10: ("LT", 2, 1), 0x11: ("GT", 2, 1), 0x12: ("SLT", 2, 1), 0x13: ("SGT", 2, 1), 0x14: ("EQ", 2, 1),
    0x15: ("ISZERO", 1, 1), 0x16: ("AND", 2, 1), 0x17: ("OR", 2, 1), 0x18: ("XOR", 2, 1), 0x19: ("NOT", 1, 1),
    0x1A: ("BYTE", 2, 1), 0x1B: ("SHL", 2, 1), 0x1C: ("SHR", 2, 1), 0x1D: ("SAR", 2, 1),
    0x20: ("SHA3", 2, 1),
    0x30: ("ADDRESS", 0, 1), 0x31: ("BALANCE", 1, 1), 0x32: ("ORIGIN", 0, 1), 0x33: ("CALLER", 0, 1),
    0x34: ("CALLVALUE", 0, 1), 0x35: ("CALLDATALOAD", 1, 1), 0x36: ("CALLDATASIZE", 0, 1),
    0x37: ("CALLDATACOPY", 3, 0), 0x38: ("CODESIZE", 0, 1), 0x39: ("CODECOPY", 3, 0), 0x3A: ("GASPRICE", 0, 1),
    0x3B: ("EXTCODESIZE", 1, 1), 0x3C: ("EXTCODECOPY", 4, 0), 0x3D: ("RETURNDATASIZE", 0, 1),
    0x3E: ("RETURNDATACOPY", 3, 0), 0x3F: ("EXTCODEHASH", 1, 1),
    0x40: ("BLOCKHASH", 1, 1), 0x41: ("COINBASE", 0, 1), 0x42: ("TIMESTAMP", 0, 1), 0x43: ("NUMBER", 0, 1),
    0x44: ("DIFFICULTY", 0, 1), 0x45: ("GASLIMIT", 0, 1), 0x46: ("CHAINID", 0, 1), 0x47: ("SELFBALANCE", 0, 1),
    0x48: ("BASEFEE", 0, 1), 0x49: ("BLOBHASH", 1, 1), 0x4A: ("BLOBBASEFEE", 0, 1),
    0x50: ("POP", 1, 0), 0x51: ("MLOAD", 1, 1), 0x52: ("MSTORE", 2, 0), 0x53: ("MSTORE8", 2, 0),
    0x54: ("SLOAD", 1, 1), 0x55: ("SSTORE", 2, 0), 0x56: ("JUMP", 1, 0), 0x57: ("JUMPI", 2, 0), 0x58: ("PC", 0, 1),
    0x59: ("MSIZE", 0, 1), 0x5A: ("GAS", 0, 1), 0x5B: ("JUMPDEST", 0, 0), 0x5C: ("TLOAD", 1, 1),
    0x5D: ("TSTORE", 2, 0), 0x5E: ("MCOPY", 3, 0), 0x5F: ("PUSH0", 0, 1),
    0xF0: ("CREATE", 3, 1), 0xF1: ("CALL", 7, 1), 0xF2: ("CALLCODE", 7, 1), 0xF3: ("RETURN", 2, 0),
    0xF4: ("DELEGATECALL", 6, 1), 0xF5: ("CREATE2", 4, 1), 0xFA: ("STATICCALL", 6, 1), 0xFD: ("REVERT", 2, 0),
    0xFE: ("INVALID", 0, 0), 0xFF: ("SELFDESTRUCT", 1, 0),
}
OPS.update({0x60 + i: (f"PUSH{i + 1}", 0, 1) for i in range(32)})
OPS.update({0x80 + i: (f"DUP{i + 1}", i + 1, i + 2) for i in range(16)})
OPS.update({0x90 + i: (f"SWAP{i + 1}", i + 2, i + 2) for i in range(16)})
OPS.update({0xA0 + i: (f"LOG{i}", i + 2, 0) for i in range(5)})

TERMINATORS = {0x00, 0x56, 0x57, 0xF3, 0xFD, 0xFE, 0xFF}
E_JUMP, E_TRUE, E_FALSE, E_FALL, E_UNKNOWN = range(5)
_STACK = 64
_MASKS = {(1 << k) - 1 for k in range(8, 257, 8)}

_VOCAB = (["STOP", "ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD", "ADDMOD", "MULMOD", "EXP", "SIGNEXTEND",
           "LT", "GT", "SLT", "SGT", "EQ", "ISZERO", "AND", "OR", "XOR", "NOT", "BYTE", "SHL", "SHR", "SAR",
           "SHA3", "ADDRESS", "BALANCE", "ORIGIN", "CALLER", "CALLVALUE", "CALLDATALOAD", "CALLDATASIZE",
           "CALLDATACOPY", "CODESIZE", "CODECOPY", "GASPRICE", "EXTCODESIZE", "EXTCODECOPY", "RETURNDATASIZE",
           "RETURNDATACOPY", "BLOCKHASH", "COINBASE", "TIMESTAMP", "NUMBER", "DIFFICULTY", "GASLIMIT",
           "POP", "MLOAD", "MSTORE", "MSTORE8", "SLOAD", "SSTORE", "JUMP", "JUMPI", "PC", "MSIZE", "GAS", "JUMPDEST",
           "LOG0", "LOG1", "LOG2", "LOG3", "LOG4", "CREATE", "CALL", "CALLCODE", "RETURN", "DELEGATECALL",
           "CREATE2", "STATICCALL", "REVERT", "INVALID", "SELFDESTRUCT"]
          + [f"PUSH{i}" for i in range(1, 33)] + [f"{k}{i}" for i in range(1, 17) for k in ("DUP", "SWAP")])
N_OPCODES = len(_VOCAB)
VOCAB_ID = {op: _VOCAB.index(name) + 1 for op, (name, _, _) in OPS.items() if name in _VOCAB}

_COUNTED = {"CALL": 0xF1, "CALLCODE": 0xF2, "DELEGATECALL": 0xF4, "STATICCALL": 0xFA, "CREATE": 0xF0,
            "CREATE2": 0xF5, "SELFDESTRUCT": 0xFF, "SSTORE": 0x55, "SLOAD": 0x54, "TIMESTAMP": 0x42,
            "NUMBER": 0x43, "ORIGIN": 0x32, "BLOCKHASH": 0x40, "JUMPI": 0x57}
_EIP1167 = bytes.fromhex("363d3d373d3d3d363d73")
_EIP1967 = bytes.fromhex("360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc")
_IFACES = {"erc20": {0x18160DDD, 0x70A08231, 0xA9059CBB, 0xDD62ED3E, 0x095EA7B3, 0x23B872DD},
           "erc721": {0x6352211E, 0x42842E0E, 0x70A08231, 0xB88D4FDE, 0x095EA7B3},
           "erc1155": {0x00FDD58E, 0x4E1273F4, 0x2EB2C2D6, 0xF242432A}}
COVARIATES = (["log_size", "has_cbor", "meta_ipfs", "meta_bzzr0", "meta_bzzr1", "solc_minor", "solc_patch",
               "log_n_selectors", "erc20", "erc721", "erc1155", "eip1167", "eip1967"]
              + [f"log_{k.lower()}" for k in _COUNTED] + ["log_logs"])


def to_bytes(code) -> bytes:
    if isinstance(code, bytes):
        return code
    return bytes.fromhex(code[2:] if code.startswith("0x") else code)


def split_metadata(full: bytes) -> tuple[bytes, bytes]:
    if len(full) > 2:
        n = int.from_bytes(full[-2:], "big")
        if 0 < n + 2 < len(full) and full[-n - 2] in (0xA1, 0xA2, 0xA3, 0xA4):
            return full[:-n - 2], full[-n - 2:-2]
    return full, b""


def runtime(code) -> bytes:
    return split_metadata(to_bytes(code))[0]


def instructions(code: bytes) -> list[tuple[int, int, int | None, int]]:
    out, i, n = [], 0, len(code)
    while i < n:
        op = code[i]
        k = op - 0x5F if 0x60 <= op <= 0x7F else 0
        imm = int.from_bytes(code[i + 1:i + 1 + k], "big") if k else (0 if op == 0x5F else None)
        out.append((i, op, imm, 1 + k))
        i += 1 + k
    return out


def opcodes(code: bytes) -> list[int]:
    return [op for _, op, _, _ in instructions(code)]


def disassembly(code: bytes) -> str:
    lines = []
    for pc, op, _, size in instructions(code):
        name = OPS.get(op, ("INVALID",))[0]
        if op == 0x5B:
            lines.append(f"{pc:x}: JUMPDEST")
        elif size > 1:
            lines.append(f"{name} 0x{code[pc + 1:pc + size].hex()}")
        else:
            lines.append(name)
    return "\n".join(lines)


def edit_distance(a: list, b: list) -> int:
    a, b = (a, b) if len(a) >= len(b) else (b, a)
    b, j = np.asarray(b), np.arange(len(b) + 1)
    prev = j.copy()
    for x in a:
        t = np.empty_like(prev)
        t[0] = prev[0] + 1
        t[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b != x))
        prev = np.minimum.accumulate(t - j) + j
    return int(prev[-1])


def fidelity(recompiled, deployed) -> float:
    a, b = opcodes(runtime(recompiled)), opcodes(runtime(deployed))
    return 1 - edit_distance(a, b) / max(len(a), len(b), 1)


def _step(st: list, op: int, imm, keep) -> None:
    if op == 0x5F or 0x60 <= op <= 0x7F:
        st.append(imm if keep(imm) else None)
    elif 0x80 <= op <= 0x8F:
        n = op - 0x7F
        st.append(st[-n] if len(st) >= n else None)
    elif 0x90 <= op <= 0x9F:
        n = op - 0x8F
        while len(st) <= n:
            st.insert(0, None)
        st[-1], st[-n - 1] = st[-n - 1], st[-1]
    elif op == 0x16:
        a = st.pop() if st else None
        b = st.pop() if st else None
        r = a & b if a is not None and b is not None else None
        st.append(r if r is not None and keep(r) else None)
    else:
        _, pops, pushes = OPS.get(op, ("INVALID", 0, 0))
        del st[max(0, len(st) - pops):]
        st.extend([None] * pushes)
    if len(st) > _STACK:
        del st[:len(st) - _STACK]


@dataclass
class CFG:
    pcs: list
    x: np.ndarray
    edges: np.ndarray
    etypes: np.ndarray
    sels: list
    fn_blocks: list
    n_blocks: int
    unresolved: float
    entries_beyond: float

    @property
    def q(self) -> list[float]:
        return [self.unresolved, float(np.log1p(self.n_blocks) / 7), float(self.n_blocks > MAX_BLOCKS),
                self.entries_beyond]


def split_blocks(ins) -> list[list]:
    end = ins[-1][0] + ins[-1][3] if ins else 0
    starts = {0}
    for pc, op, _, size in ins:
        if op == 0x5B:
            starts.add(pc)
        if op in TERMINATORS or op not in OPS:
            starts.add(pc + size)
    starts = sorted(s for s in starts if s < end)
    blocks, j = [], 0
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else end
        bl = []
        while j < len(ins) and ins[j][0] < e:
            if ins[j][0] >= s:
                bl.append(ins[j])
            j += 1
        if bl:
            blocks.append(bl)
    return blocks


def _dispatcher(blocks, at_dest) -> tuple[dict, set]:
    entries, compare = {}, set()
    for b, bl in enumerate(blocks):
        if len(bl) < 3 or bl[-1][1] != 0x57 or not 0x60 <= bl[-2][1] <= 0x7F or bl[-2][2] not in at_dest:
            continue
        for k in range(len(bl) - 3, -1, -1):
            if bl[k][1] != 0x14:
                continue
            for kk in range(k - 1, max(-1, k - 4), -1):
                if bl[kk][1] == 0x63 and bl[kk][2] != 0xFFFFFFFF:
                    entries.setdefault(bl[kk][2], at_dest[bl[-2][2]])
                    compare.add(b)
                    break
            break
    return entries, compare


def build_cfg(code, max_blocks: int = MAX_BLOCKS, max_contexts: int = 8, max_states: int = 50_000) -> CFG:
    code = runtime(code)
    ins = instructions(code)
    blocks = split_blocks(ins)
    at_dest = {bl[0][0]: i for i, bl in enumerate(blocks) if bl[0][1] == 0x5B}
    keep = lambda v: v is not None and (v in at_dest or v in _MASKS)
    edges, unknown, resolved = set(), set(), set()
    seen, work, states = defaultdict(set), deque([(0, ())]), 0
    while work and states < max_states and blocks:
        b, st0 = work.popleft()
        if st0 in seen[b] or len(seen[b]) >= max_contexts:
            continue
        seen[b].add(st0)
        states += 1
        st, target, bl = list(st0), None, blocks[b]
        for k, (_, op, imm, _) in enumerate(bl):
            if k == len(bl) - 1 and op in (0x56, 0x57):
                target = st[-1] if st else None
            _step(st, op, imm, keep)
        tail, nxt = bl[-1][1], b + 1 if b + 1 < len(blocks) else None
        succ = []
        if tail in (0x56, 0x57):
            if target in at_dest:
                succ.append((at_dest[target], E_JUMP if tail == 0x56 else E_TRUE))
                resolved.add(b)
            else:
                unknown.add(b)
            if tail == 0x57 and nxt is not None:
                succ.append((nxt, E_FALSE))
        elif tail not in TERMINATORS and tail in OPS and nxt is not None:
            succ.append((nxt, E_FALL))
        out = tuple(st)
        for t, et in succ:
            edges.add((b, t, et))
            work.append((t, out))
    jumps = sum(bl[-1][1] in (0x56, 0x57) for bl in blocks)
    never = unknown - resolved
    edges |= {(b, b, E_UNKNOWN) for b in never}
    entries, compare = _dispatcher(blocks, at_dest)
    adj = defaultdict(list)
    for u, v, _ in edges:
        adj[u].append(v)
    sels, fn_blocks = [], []
    for sel, e in sorted(entries.items(), key=lambda kv: (kv[1], kv[0])):
        reach, stack = {e}, [e]
        while stack:
            for v in adj[stack.pop()]:
                if v not in reach and v not in compare:
                    reach.add(v)
                    stack.append(v)
        sels.append(int(sel))
        fn_blocks.append(sorted(v for v in reach if v < max_blocks))
    n = min(len(blocks), max_blocks)
    x = np.zeros((n, BLOCK_OPS), np.uint8)
    for i, bl in enumerate(blocks[:n]):
        ids = [VOCAB_ID.get(op, 0) for _, op, _, _ in bl[:BLOCK_OPS]]
        x[i, :len(ids)] = ids
    kept = sorted((u, v, t) for u, v, t in edges if u < n and v < n)
    return CFG(pcs=[bl[0][0] for bl in blocks[:n]], x=x,
               edges=np.asarray([(u, v) for u, v, _ in kept], np.int16).reshape(-1, 2),
               etypes=np.asarray([t for _, _, t in kept], np.uint8), sels=sels, fn_blocks=fn_blocks,
               n_blocks=len(blocks), unresolved=len(never) / max(jumps, 1),
               entries_beyond=float(np.mean([e >= max_blocks for e in entries.values()])) if entries else 0.0)


def selector_agreement(d_sels, g_sels) -> tuple[float, float, float]:
    d, g = {s for s in d_sels if s is not None and s >= 0}, set(g_sels)
    both = len(d & g)
    return both / max(len(d | g), 1), both / max(len(d), 1) if d else 0.0, both / len(g) if g else 0.0


def covariates(code) -> np.ndarray:
    full = to_bytes(code)
    body, meta = split_metadata(full)
    counts, sels = np.zeros(256, np.int64), set()
    i, n = 0, len(body)
    while i < n:
        op = body[i]
        counts[op] += 1
        if op == 0x63 and i + 5 < n:
            nxt = body[i + 5]
            if nxt == 0x14 or (0x80 <= nxt <= 0x82 and i + 6 < n and body[i + 6] == 0x14):
                sels.add(int.from_bytes(body[i + 1:i + 5], "big"))
        i += 1 + (op - 0x5F if 0x60 <= op <= 0x7F else 0)
    solc = meta.find(b"dsolcC")
    minor, patch = (meta[solc + 7], meta[solc + 8]) if solc >= 0 and solc + 8 < len(meta) else (-1, -1)
    z = [np.log1p(len(body)), float(len(meta) > 0), float(b"dipfs" in meta), float(b"bzzr0" in meta),
         float(b"bzzr1" in meta), float(minor), float(patch), np.log1p(len(sels))]
    z += [float(s <= sels) for s in _IFACES.values()]
    z += [float(full.startswith(_EIP1167)), float(_EIP1967 in full)]
    z += [np.log1p(counts[op]) for op in _COUNTED.values()] + [np.log1p(counts[0xA0:0xA5].sum())]
    return np.asarray(z, np.float32).round(4)
