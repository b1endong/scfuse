import json
import os
import random
from datetime import datetime
from pathlib import Path

from scfuse.prep.common import Raw, parser, write_jsonl
from scfuse.utils.solc import version

GOLD_PRIORITY = {"gold_test": 0, "gold_val": 1, "gold_anchor": 2}


def sanctuary(raw: Raw) -> tuple[dict, dict]:
    paths, meta = {}, {}
    for sub in sorted(os.listdir(raw.sanctuary)):
        d = raw.sanctuary / sub
        if d.is_dir():
            for fn in os.listdir(d):
                if fn.endswith(".sol"):
                    paths.setdefault("0x" + fn.split("_", 1)[0].lower(), str(d / fn))
    with open(raw.sanctuary / "contracts.json") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                date = datetime.strptime(r.get("date") or "", "%m/%d/%Y").date().isoformat()
            except ValueError:
                date = None
            meta[r["address"].lower()] = {"date": date, "solc": version(r.get("compiler")),
                                          "settings": r.get("settings") or ""}
    return paths, meta


def build(raw: Raw, splits: Path, weak_sample: int | None, seed: int = 0) -> tuple[list[dict], dict]:
    paths, meta = sanctuary(raw)
    clusters = {r["address"]: r["cluster_id"] for r in json.load(open(raw.dedup))} if raw.dedup.exists() else {}
    rows, gold, dropped = [], {}, {}
    for split in ("train", "test", "val", "gold_anchor"):
        recs = json.load(open(splits / f"{split}.json"))
        weak = [r for r in recs if r["origin"] == "weak"]
        if weak_sample and split in ("train", "test"):
            n = weak_sample if split == "train" else weak_sample // 4
            weak = random.Random(seed).sample(weak, min(len(weak), n))
        if split != "val":
            for r in weak:
                a = r["address"].lower()
                m = meta.get(a, {})
                rows.append({"id": a, "kind": "weak", "split": split, "sol_path": paths.get(a), "solc": m.get("solc"),
                             "settings": m.get("settings", ""), "date": m.get("date"), "cluster": clusters.get(a),
                             "y_gold": None, "audited_all": 0})
        for r in recs:
            if r["origin"] == "weak":
                continue
            cid = r["contract_id"]
            row = {"id": cid, "kind": cid.split("_", 1)[0], "split": r["origin"], "sol_path": r["source_path"],
                   "solc": version(r.get("solc")), "settings": "", "date": None, "cluster": None,
                   "y_gold": r["label"], "audited_all": 0}
            if cid in gold:
                keep, drop = sorted([gold[cid], row], key=lambda x: GOLD_PRIORITY[x["split"]])
                keep["y_gold"] = [max(a, b) for a, b in zip(keep["y_gold"], drop["y_gold"])]
                dropped[f"{cid}@{drop['split']}"] = f"same contract as the {keep['split']} copy; labels merged"
                gold[cid] = keep
            else:
                gold[cid] = row
    return rows + list(gold.values()), dropped


def main():
    ap = parser("Contract index from the frozen splits")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--weak-sample", type=int, default=None)
    args = ap.parse_args()
    cache = Path(args.cache)
    rows, dropped = build(Raw(args.raw), Path(args.splits), args.weak_sample)
    write_jsonl(cache / "index.jsonl", rows)
    json.dump(dropped, open(cache / "index_dropped.json", "w"), indent=1)
    by = {}
    for r in rows:
        by[r["split"]] = by.get(r["split"], 0) + 1
    print(f"index: {len(rows)} contracts {by}; without source: {sum(r['sol_path'] is None for r in rows)}")


if __name__ == "__main__":
    main()
