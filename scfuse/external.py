import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scfuse.config import load_config, run_name
from scfuse.data import Collate, ContractDataset, load_packed
from scfuse.labels import ANALYZERS, load_label_map
from scfuse.metrics import FINE_GRID, average_precision, ece, stratified_auroc, tune_thresholds
from scfuse.prep.lifted import units as lift_units
from scfuse.prep.pack import text_arrays
from scfuse.protocols import Override
from scfuse.schema import ARRAYS, CHARS_SCALE, Q_LAYOUT, Q_LAYOUT_V25, UNITS_SCALE
from scfuse.stats import project_bootstrap
from scfuse.utils.ast import LABELS
from scfuse.utils.evm import build_cfg, fidelity, runtime, selector_agreement
from scfuse.utils.solc import compile_lifted


def prepare(args):
    from transformers import AutoTokenizer
    from scfuse.prep.encode import encode_view
    lm = load_label_map()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    out, cache = Path(args.out), Path(args.out) / "cache"
    (cache / "packed").mkdir(parents=True, exist_ok=True)
    layout = Q_LAYOUT if args.layout == "scfuse" else Q_LAYOUT_V25
    handles = {k: open(cache / "packed" / f"{k}.bin", "wb") for k in ARRAYS}
    counts, metas, labels = dict.fromkeys(ARRAYS, 0), [], {}
    for line in open(args.cohort):
        r = json.loads(line)
        g = build_cfg(r["bytecode"])
        path = Path(args.lifted) / f"{r['id']}.sol"
        text = path.read_text(errors="replace") if path.exists() else ""
        us = lift_units(text) if text else []
        ids = tok([u["text"] for u in us], add_special_tokens=False)["input_ids"] if us else []
        arr = {f"D_{k}": a for k, a in text_arrays(ids, [u["sel"] for u in us]).items()}
        arr.update({f"S_{k}": a for k, a in text_arrays([], []).items()})
        arr.update(G_x=g.x, G_edge=g.edges, G_etype=g.etypes, G_fnsel=np.asarray(g.sels, np.int64),
                   G_mem=np.asarray([[b, f] for f, bl in enumerate(g.fn_blocks) for b in bl], np.int32).reshape(-1, 2),
                   G_pc=np.asarray(g.pcs, np.int32), A_node=np.zeros((0, 3), np.int16), A_unit=np.zeros((0, 3), np.int64),
                   SP_pair=np.zeros((0, 2), np.int32), SP_span=np.zeros((0, 3), np.int32))
        compiled = compile_lifted(text, r.get("solc")) if text else None
        agree, _, cover = selector_agreement([u["sel"] for u in us], g.sels)
        n_u, n_c = np.log1p(len(us)) / UNITS_SCALE["D"], np.log1p(len(text)) / CHARS_SCALE
        q_D = ([float(compiled is not None), fidelity(compiled, r["bytecode"]) if compiled else 0.0, agree, n_u, n_c]
               if args.layout == "scfuse" else [0.0, n_u, cover, n_c])
        m = {"id": r["id"], "kind": "external", "split": "external", "date": None, "cluster": None, "m_S": 0,
             "m_G": int(len(g.pcs) > 0), "m_D": int(len(us) > 0), "m_A": 0, "y_gold": [0] * lm.K, "audited_all": 0,
             "q_S": [0.0] * len(layout["S"]), "q_G": g.q, "q_D": q_D, "q_A": [0.0] * 4, "e": 0.0, "w": 1.0,
             "project": r["project"], "size": len(runtime(r["bytecode"]))}
        for a in ANALYZERS:
            m[f"y_{a}"], m[f"obs_{a}"] = [0] * lm.K, [0] * lm.K
        for k, a in arr.items():
            m[f"{k}_0"], m[f"{k}_n"] = counts[k], len(a)
            handles[k].write(np.ascontiguousarray(a, dtype=ARRAYS[k][0]).tobytes())
            counts[k] += len(a)
        metas.append(m)
        labels[r["id"]] = r["labels"]
    for h in handles.values():
        h.close()
    pd.DataFrame(metas).to_parquet(cache / "packed" / "meta.parquet")
    json.dump({k: {"dtype": ARRAYS[k][0], "width": ARRAYS[k][1], "rows": counts[k]} for k in ARRAYS},
              open(cache / "packed" / "arrays.json", "w"), indent=1)
    json.dump({"version": 3 if args.layout == "scfuse" else 2.5, "q": layout, "ast_vocab": len(LABELS)},
              open(cache / "packed" / "layout.json", "w"), indent=1)
    json.dump(labels, open(out / "labels.json", "w"))
    feats = cache / "feats" / args.backbone.split("/")[-1]
    feats.mkdir(parents=True, exist_ok=True)
    encode_view(load_packed(cache), "D", feats, args.backbone, args.batch_tokens, args.device)
    print(f"prepared {len(metas)} external contracts -> {cache}")


def evaluate(args):
    from scfuse.train import datasets, load_checkpoint, predict
    cfg = load_config(args.config, args.overrides)
    lm = load_label_map()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prep = Path(args.prepared)
    out = prep / "eval" / cfg["run"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    ext = None
    for seed in cfg["seeds"]:
        path = Path(cfg["run"]["ckpt"]) / run_name(cfg, seed) / "best.pt"
        if not path.exists():
            continue
        model, ck = load_checkpoint(path, dev, lm.K)
        mc = ck["cfg"]
        p, _, ds = datasets(mc, train_caps=False)
        bad = np.isnan(ds.e) | np.isnan(ds.w)
        ds.e[bad], ds.w[bad] = ck.get("e_bar", 0.5), 1.0
        val = np.flatnonzero(p.meta.split.to_numpy() == "gold_val")
        bf16 = mc["train"]["bf16"] and dev.type == "cuda"
        if ext is None:
            mode = "cached" if mc["model"]["regime"] == "cached" else "tokens"
            ext = ContractDataset(load_packed(prep / "cache"), mode, prep / "cache" / "feats" / mc["data"]["feats"])
        rows = np.arange(len(ext))

        def run(data, base, idx, mode):
            loader = torch.utils.data.DataLoader(data, sampler=idx.tolist(), batch_size=mc["train"]["eval_batch"],
                                                 collate_fn=Collate(base))
            return predict(model, loader, dev, bf16, mode)["P"]

        no_d = lambda d, idx: Override(d, m={int(i): 0.0 for i in idx})
        tau_p4 = tune_thresholds(run(ds, ds, val, "bytecode"), ds.y_gold[val], FINE_GRID)
        tau_g = tune_thresholds(run(no_d(ds, val), ds, val, "bytecode"), ds.y_gold[val], FINE_GRID)
        np.savez_compressed(out / f"s{seed}.npz", P_P4=run(ext, ext, rows, "bytecode"),
                            P_G=run(no_d(ext, rows), ext, rows, "bytecode"), tau_P4=tau_p4, tau_G=tau_g)
        print(f"{cfg['run']['name']} seed {seed}: external predictions -> {out}")


def _scores(prep: Path, run: str):
    files = sorted((prep / "eval" / run).glob("s*.npz"))
    return [np.load(f) for f in files]


def report(args):
    lm = load_label_map()
    prep = Path(args.prepared)
    meta = pd.read_parquet(prep / "cache" / "packed" / "meta.parquet")
    labels = json.load(open(prep / "labels.json"))
    L = np.full((len(meta), lm.K), np.nan)
    for r, cid in enumerate(meta.id):
        for c, v in labels[cid].items():
            if v is not None and c in lm.codes:
                L[r, lm.codes.index(c)] = float(v)
    classes = list(lm.external_idx)
    projects, size = meta.project.to_numpy(), meta["size"].to_numpy().astype(float)
    strata = np.digitize(size, np.quantile(size, [0.2, 0.4, 0.6, 0.8]))
    ref = [z["P_P4"] for z in _scores(prep, args.ref)] if args.ref else None
    res = {}
    baselines = {"runtime_size": [np.repeat(size[:, None], lm.K, 1)]}
    for name in list(args.runs) + list(baselines):
        if name in baselines:
            p4, gonly, taus = baselines[name], baselines[name], None
        else:
            zs = _scores(prep, name)
            if not zs:
                continue
            p4, gonly, taus = [z["P_P4"] for z in zs], [z["P_G"] for z in zs], [z["tau_P4"] for z in zs]
        row = {"seeds": len(p4), "auroc": project_bootstrap(p4, None, L, classes, projects, args.reps),
               "auroc_g_only": project_bootstrap(gonly, None, L, classes, projects, args.reps)}
        strat, ap = [], []
        for k in classes:
            known = ~np.isnan(L[:, k])
            if (L[known, k] > 0).any():
                strat.append(np.mean([stratified_auroc(s[known, k], L[known, k] > 0, strata[known]) for s in p4]))
                ap.append(np.mean([average_precision(s[known, k], L[known, k]) for s in p4]))
        row["auroc_size_strata"], row["auprc"] = float(np.nanmean(strat)), float(np.nanmean(ap))
        if ref is not None and name != args.ref:
            row["delta_auroc_vs_ref"] = project_bootstrap(p4, ref, L, classes, projects, args.reps)
        if taus is not None:
            rec, f1, ec = [], [], []
            for s, t in zip(p4, taus):
                pred = s >= t
                for k in classes:
                    known = ~np.isnan(L[:, k])
                    pos = known & (L[:, k] > 0)
                    if pos.any():
                        tp = (pred[pos, k]).sum()
                        fp = (pred[known & (L[:, k] == 0), k]).sum()
                        rec.append(tp / pos.sum())
                        f1.append(2 * tp / max(2 * tp + fp + (pos.sum() - tp), 1))
                known = ~np.isnan(L[:, classes])
                ec.append(ece(s[:, classes][known][:, None], L[:, classes][known][:, None]))
            row.update(recall=float(np.mean(rec)), macro_f1=float(np.mean(f1)), ece=float(np.mean(ec)))
        res[name] = row
    known = ~np.isnan(L)
    f1_all = []
    for k in classes:
        pos = known[:, k] & (L[:, k] > 0)
        if pos.any():
            f1_all.append(2 * pos.sum() / (pos.sum() + known[:, k].sum()))
    res["flag_every_contract"] = {"recall": 1.0, "macro_f1": float(np.mean(f1_all))}
    res["random"] = {"auroc": 0.5}
    res["cohort"] = {"contracts": int(len(meta)), "projects": int(len(np.unique(projects))),
                     "positives": {lm.codes[k]: int(np.nansum(L[:, k] > 0)) for k in classes}}
    out = prep / "report.json"
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v.get("auroc") for k, v in res.items() if isinstance(v, dict)}, indent=1))


def main():
    ap = argparse.ArgumentParser(description="P8: external audited cohort, bytecode only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--cohort", required=True)
    a.add_argument("--lifted", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--layout", choices=("scfuse", "v25"), default="scfuse")
    a.add_argument("--backbone", default="microsoft/unixcoder-base")
    a.add_argument("--batch-tokens", type=int, default=65_536)
    a.add_argument("--device", default="cuda")
    b = sub.add_parser("evaluate")
    b.add_argument("--prepared", required=True)
    b.add_argument("--config", default="configs/default.yaml")
    b.add_argument("overrides", nargs="*")
    c = sub.add_parser("report")
    c.add_argument("--prepared", required=True)
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--ref", default="scfuse")
    c.add_argument("--reps", type=int, default=10_000)
    args = ap.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
