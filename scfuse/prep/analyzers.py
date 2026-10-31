import json
import os
import random
import resource
import subprocess
import tempfile
from pathlib import Path

from scfuse.prep.common import Bytecodes, Raw, load_index, parser, pool_map, write_jsonl
from scfuse.utils.evm import runtime
from scfuse.utils.solc import binary

TIMEOUT = 60
MEMORY = 6 << 30
_G = {}


def _init(raw_root: str):
    raw = Raw(raw_root)
    _G.update(raw=raw, bc=Bytecodes(raw))


def _limit():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))


def run_slither(path: Path, solc: str | None) -> tuple[str, list]:
    exe = binary(solc) if solc else None
    cmd = ["slither", str(path), "--json", "-"] + (["--solc", exe] if exe else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, preexec_fn=_limit,
                           cwd=path.parent)
        out = json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return "timeout", []
    except json.JSONDecodeError:
        return "error", []
    if not out.get("success"):
        return "error", []
    return "ok", sorted({d["check"] for d in out.get("results", {}).get("detectors", [])})


def run_mythril(code: str) -> tuple[str, list]:
    with tempfile.NamedTemporaryFile("w", suffix=".bin", delete=False) as f:
        f.write(runtime(code).hex())
        path = f.name
    try:
        r = subprocess.run(["myth", "analyze", "--bin-runtime", "-f", path, "-o", "json", "--execution-timeout",
                            str(TIMEOUT)], capture_output=True, text=True, timeout=TIMEOUT * 3, preexec_fn=_limit)
        out = json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return "timeout", []
    except json.JSONDecodeError:
        return "error", []
    finally:
        os.unlink(path)
    if not out.get("success", True) and not out.get("issues"):
        return "error", []
    return "ok", sorted({int(i["swc-id"]) for i in out.get("issues", []) if str(i.get("swc-id", "")).isdigit()})


def job(item: tuple[str, dict]) -> dict:
    analyzer, row = item
    if analyzer == "slither":
        path = _G["raw"].path(row.get("sol_path"))
        status, found = run_slither(path, row.get("solc")) if path and path.exists() else ("no_input", [])
    else:
        code = _G["bc"].get(row["id"])
        status, found = run_mythril(code) if code else ("no_input", [])
    return {"id": row["id"], "analyzer": analyzer, "status": status, "found": found}


def main():
    ap = parser("Analyzer labels: Slither (reads S) and Mythril (reads G)")
    ap.add_argument("mode", choices=("gold", "weak", "import"))
    ap.add_argument("--weak-n", type=int, default=4999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cache = Path(args.cache)
    rows = load_index(cache)
    out = cache / "analyzers"
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "import":
        recs = []
        for line in open(Raw(args.raw).weak_slither):
            r = json.loads(line)
            recs.append({"id": r["address"].lower(), "analyzer": "slither", "status": r["slither_status"],
                         "found": r["detectors"]})
        n = write_jsonl(out / "slither_weak.jsonl", recs)
        print(f"imported {n} Slither results")
        return
    if args.mode == "gold":
        gold = [r for r in rows if r["kind"] != "weak"]
        items = [("slither", r) for r in gold] + [("mythril", r) for r in gold]
        name = "gold"
    else:
        excluded = json.load(open(cache / "excluded.json")) if (cache / "excluded.json").exists() else {}
        train = sorted(r["id"] for r in rows if r["kind"] == "weak" and r["split"] == "train" and r["id"] not in excluded)
        subset = sorted(random.Random(args.seed).sample(train, min(args.weak_n, len(train))))
        (cache / "subsets").mkdir(exist_ok=True)
        json.dump(subset, open(cache / "subsets" / "mythril5k.json", "w"))
        keep = set(subset)
        items = [("mythril", r) for r in rows if r["id"] in keep]
        name = "mythril_weak"
    n = write_jsonl(out / f"{name}.jsonl", pool_map(job, items, args.workers, f"analyzers {args.mode}", 1, _init,
                                                    (args.raw,)))
    print(f"{name}: {n} analyses")


if __name__ == "__main__":
    main()
