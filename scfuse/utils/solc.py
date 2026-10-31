import json
import os
import re
import subprocess
from pathlib import Path

from scfuse.utils.evm import runtime

ARTIFACTS = Path(os.environ.get("SOLC_ARTIFACTS", Path.home() / ".solc-select" / "artifacts"))
OUTPUTS = ["evm.deployedBytecode.object", "evm.deployedBytecode.sourceMap",
           "evm.deployedBytecode.immutableReferences", "evm.methodIdentifiers"]


def version(v) -> str | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", v or "")
    return ".".join(m.groups()) if m else None


def vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def binary(ver: str, install: bool = False) -> str | None:
    path = ARTIFACTS / f"solc-{ver}" / f"solc-{ver}"
    if not path.exists() and install:
        subprocess.run(["solc-select", "install", ver], capture_output=True)
    return str(path) if path.exists() else None


def standard_json(ver: str, sources: dict[str, str], optimize: bool, runs: int = 200, ast: bool = True,
                  timeout: int = 120, install: bool = False) -> dict | None:
    if not ver or vtuple(ver) < (0, 4, 11):
        return None
    exe = binary(ver, install)
    if exe is None:
        return None
    req = {"language": "Solidity", "sources": {k: {"content": v} for k, v in sources.items()},
           "settings": {"optimizer": {"enabled": optimize, "runs": runs},
                        "outputSelection": {"*": {"*": OUTPUTS, "": ["ast", "legacyAST"] if ast else []}}}}
    try:
        r = subprocess.run([exe, "--standard-json"], input=json.dumps(req), capture_output=True, text=True,
                           timeout=timeout)
        out = json.loads(r.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    if any(e.get("severity") == "error" for e in out.get("errors", [])):
        return None
    return out


def deployed(c: dict) -> tuple[str, str, dict]:
    d = c.get("evm", {}).get("deployedBytecode", {})
    return d.get("object") or "", d.get("sourceMap") or "", d.get("immutableReferences") or {}


def same_runtime(compiled_hex: str, chain_hex: str, immutables: dict) -> bool:
    if not compiled_hex or "__" in compiled_hex:
        return False
    try:
        a, b = bytearray(runtime(compiled_hex)), bytearray(runtime(chain_hex))
    except ValueError:
        return False
    if len(a) != len(b):
        return False
    for refs in immutables.values():
        for r in refs:
            a[r["start"]:r["start"] + r["length"]] = b[r["start"]:r["start"] + r["length"]]
    return a == b


def parse_srcmap(srcmap: str) -> list[tuple[int, int, int]]:
    s = n = f = -1
    out = []
    for item in srcmap.split(";") if srcmap else []:
        parts = item.split(":")
        s = int(parts[0]) if parts[0] else s
        n = int(parts[1]) if len(parts) > 1 and parts[1] else n
        f = int(parts[2]) if len(parts) > 2 and parts[2] else f
        out.append((s, n, f))
    return out


def compile_lifted(code: str, ver: str, timeout: int = 120) -> str | None:
    if not ver:
        return None
    body = re.sub(r"pragma\s+solidity[^;]*;", "", code)
    out = standard_json(ver, {"D.sol": f"pragma solidity {ver};\n{body}"}, True, 200, ast=False, timeout=timeout)
    if out is None:
        return None
    bins = [deployed(c)[0] for cs in out.get("contracts", {}).values() for c in cs.values()]
    bins = [b for b in bins if b and "__" not in b]
    return "0x" + max(bins, key=len) if bins else None
