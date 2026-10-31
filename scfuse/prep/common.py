import argparse
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

_ID = re.compile(rb'"(?:id|address)": "([^"]+)"')


class Raw:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        d = self.root / "data"
        self.sanctuary = d / "ethereum" / "contracts" / "mainnet"
        self.bytecode = d / "interim" / "contracts_with_bytecode.jsonl"
        self.dedup = d / "interim" / "dedup_clusters.json"
        self.heimdall = d / "processed" / "lifted"
        self.weak_slither = d / "processed" / "labels" / "weak_labels.jsonl"
        self.gold_bytecode = d / "cache" / "gold_bytecode.json"
        self.pool = d / "raw" / "deployment_pool.parquet"
        self.etherscan = d / "raw" / "deployment_population_etherscan.jsonl"

    def path(self, rel: str | None) -> Path | None:
        return None if rel is None else (Path(rel) if os.path.isabs(rel) else self.root / rel)


def parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--cache", default=os.environ.get("SCFUSE_CACHE", "data/scfuse"))
    ap.add_argument("--raw", default=os.environ.get("SCFUSE_RAW", ".."))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return ap


def load_index(cache: str | Path) -> list[dict]:
    return [json.loads(line) for line in open(Path(cache) / "index.jsonl")]


def line_offsets(path: Path) -> dict[str, int]:
    out, pos = {}, 0
    with open(path, "rb") as f:
        for line in f:
            m = _ID.search(line)
            if m:
                out.setdefault(m.group(1).decode().lower() if m.group(1).startswith(b"0x") else m.group(1).decode(),
                               pos)
            pos += len(line)
    return out


def read_at(path: Path, off: int | None) -> dict | None:
    if off is None:
        return None
    with open(path, "rb") as f:
        f.seek(off)
        return json.loads(f.readline())


class Records:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.off = line_offsets(self.path) if self.path.exists() else {}

    def get(self, key: str) -> dict | None:
        return read_at(self.path, self.off.get(key))

    def __contains__(self, key: str) -> bool:
        return key in self.off


class Bytecodes:
    def __init__(self, raw: Raw):
        self.weak = Records(raw.bytecode)
        self.gold = {k: v["bc"] if isinstance(v, dict) else v
                     for k, v in json.load(open(raw.gold_bytecode)).items()} if raw.gold_bytecode.exists() else {}

    def get(self, cid: str) -> str | None:
        if cid in self.gold:
            return self.gold[cid]
        r = self.weak.get(cid.lower())
        return r.get("bytecode") if r and r.get("fetch_status", "ok") == "ok" else None


def pool_map(fn, items, workers: int, desc: str, chunksize: int = 16, init=None, initargs=(), ordered: bool = False):
    items = list(items)
    if workers <= 1:
        if init:
            init(*initargs)
        yield from tqdm(map(fn, items), total=len(items), desc=desc)
        return
    with Pool(workers, initializer=init, initargs=initargs) as pool:
        it = pool.imap if ordered else pool.imap_unordered
        yield from tqdm(it(fn, items, chunksize=chunksize), total=len(items), desc=desc)


def write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n
