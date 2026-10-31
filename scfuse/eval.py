import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from scfuse.config import load_config, run_name
from scfuse.data import Collate
from scfuse.labels import ANALYZERS, load_label_map
from scfuse.metrics import (FINE_GRID, auprc, deferral, ece, f1_per_class, micro_f1, select_umax, selective,
                             tune_thresholds, uncertainty_quintiles)
from scfuse.model.certify import certified, inclusion_bound, interval
from scfuse.protocols import (CORRUPTIONS, MASKS, P6_KINDS, RHOS, Override, broken_share, corrupt, gate_threshold,
                               gated, load_mutations, mutated)
from scfuse.schema import SOURCES, VI
from scfuse.stats import bootstrap_ci
from scfuse.train import datasets, load_checkpoint, predict

ALL = ("P1", "P2", "P3", "P4", "P5_replaced", "P5_half", "P5_gate", "P5prime", "P6")
Q_D = (VI["D"], len(SOURCES) - 1)
KEYS = ("P", "u", "upsilon", "omega", "m_src")
SUBSETS = ("all", "solidifi", "smartbugs")


def host_of(cid: str) -> str:
    return "_".join(cid.split("_")[-2:]) if cid.startswith("solidifi_") else cid


def nanmean(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(x, float), axis)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 5)
    if isinstance(x, (np.integer, np.bool_)):
        return int(x)
    return x


def mean_dicts(ds: list):
    if not ds:
        return {}
    if isinstance(ds[0], dict):
        return {k: mean_dicts([d[k] for d in ds if k in d]) for k in ds[0]}
    if isinstance(ds[0], (list, np.ndarray)):
        return nanmean(np.asarray(ds, float), 0).tolist()
    return float(nanmean([np.nan if v is None else v for v in ds]))


def scores(P, u, pred, y, ev, subsets: dict) -> dict:
    out = {}
    for name, rows in subsets.items():
        if not rows.any():
            continue
        f1 = f1_per_class(pred[rows], y[rows])
        aurc, r80, _ = selective(1 - u[rows][:, ev].mean(1), pred[rows], y[rows], ev)
        out[name] = {"macro_f1": float(nanmean(f1[ev])), "micro_f1": micro_f1(pred[rows], y[rows], ev),
                     "auprc": auprc(P[rows], y[rows], ev), "ece": ece(P[rows], y[rows]), "aurc": aurc,
                     "risk_at_80": r80, "f1": f1.tolist()}
    return out


def diagnostics(o: dict, model, tau, ev) -> dict:
    r, s, om, m = o["r"], o["s"], o["omega"], o["m_src"]
    n = (m * om)[..., None] * (r + s)
    den = n[:, VI["D"]] + n[:, -1]
    a = model.a.cpu().numpy()
    delta, P_minus = inclusion_bound(r, s, om, m, a, model.W, Q_D)
    realized = np.abs(o["P"] - P_minus)
    lo, hi = interval(r, s, om, m, a, model.W, model.Emax, Q_D)
    lo2, hi2 = interval(r, s, om, m, a, model.W, model.Emax, Q_D, m * om)
    pct = lambda x: np.percentile(x, [5, 25, 50, 75, 95]).tolist()
    hasD = m[:, VI["D"]] > 0
    return {"omega": ((om * m).sum(0) / np.maximum(m.sum(0), 1)).tolist(),
            "omega_D": float(om[hasD, VI["D"]].mean()) if hasD.any() else float("nan"),
            "via_J": float((n[:, -1][den > 0] / den[den > 0]).mean()) if (den > 0).any() else float("nan"),
            "within_bound": float((realized <= delta + 1e-6).mean()), "bound_q": pct(delta),
            "realized_q": pct(realized), "certified": float(certified(lo, hi, tau)[:, ev].mean()),
            "certified_omega": float(certified(lo2, hi2, tau)[:, ev].mean())}


class Runner:
    def __init__(self, model, ds, rows: dict, dev, bf16: bool, batch: int, workers: int):
        self.model, self.ds, self.rows, self.dev, self.bf16 = model, ds, rows, dev, bf16
        self.batch, self.workers = batch, workers
        self.keys = KEYS + (("r", "s") if model.head == "evidential" else ())

    def __call__(self, split: str, mode: str = "full", e_bar: float = 0.5, seed: int = 0, data=None) -> dict:
        data = data or self.ds
        loader = torch.utils.data.DataLoader(data, sampler=self.rows[split].tolist(), batch_size=self.batch,
                                             collate_fn=Collate(self.ds), num_workers=self.workers)
        return predict(self.model, loader, self.dev, self.bf16, mode, e_bar, seed, self.keys)


def corrupted(run: Runner, o: dict, tau, ref: dict, ev, subsets: dict, valid=None) -> dict:
    pred = o["P"] >= tau
    rows = np.ones(len(pred), bool) if valid is None else valid
    res = scores(o["P"], o["u"], pred, o["y"], ev, subsets)
    res["flip_rate"] = float((pred[rows][:, ev] != ref["pred"][rows][:, ev]).mean()) if rows.any() else float("nan")
    res["mean_abs_dP"] = float(np.abs(o["P"][rows] - ref["P"][rows]).mean()) if rows.any() else float("nan")
    if run.model.head == "evidential":
        res["diag"] = diagnostics(o, run.model, tau, ev)
    return res, pred


def evaluate_seed(run: Runner, protocols: list, ec: dict, lm, cache: Path, feats_dir: Path) -> tuple[dict, dict]:
    ds, ev, rows = run.ds, list(lm.eval_idx), run.rows
    kind = ds.p.meta.kind.to_numpy()[rows["test"]]
    subsets = {"all": np.ones(len(kind), bool), "solidifi": kind == "solidifi", "smartbugs": kind == "smartbugs"}
    model = run.model
    res, preds = {}, {}
    ov, ot = run("val"), run("test")
    tau = tune_thresholds(ov["P"], ov["y"], FINE_GRID)
    y, pred1 = ot["y"], ot["P"] >= tau
    ref = {"pred": pred1, "P": ot["P"]}
    p1 = scores(ot["P"], ot["u"], pred1, y, ev, subsets)
    floor = model.W / (model.W + 2 * len(SOURCES) * model.Emax) if model.head == "evidential" else 0.0
    u_max = select_umax(ov["upsilon"], ov["P"] >= tau, ov["y"], ev, ec["min_coverage"], floor)
    p1["deferral"] = {"u_max": u_max, "floor": floor, **deferral(ot["upsilon"], pred1, y, ev, u_max)}
    p1["quintile_error"] = uncertainty_quintiles(ot["u"][:, ev].mean(1), pred1, y, ev)
    _, _, curve = selective(1 - ot["u"][:, ev].mean(1), pred1, y, ev)
    p1["risk_curve"] = curve.tolist()
    p1["tau"] = tau.tolist()
    if model.head == "evidential":
        p1["diag"] = diagnostics(ot, model, tau, ev)
    res["P1"] = p1
    preds.update(P1_pred=pred1, P1_P=ot["P"], P1_u=ot["u"], P1_upsilon=ot["upsilon"], y=y,
                 kind=kind, host=np.asarray([host_of(c) for c in ds.p.meta.id.to_numpy()[rows["test"]]]),
                 idx=ot["idx"])
    for name in ("P2", "P3", "P4"):
        if name not in protocols:
            continue
        mode, draws = MASKS[name]
        per, stack = [], []
        for d in range(draws):
            ov_ = run("val", mode, float(ds.e[rows["val"]].mean()), 1000 + d)
            ot_ = run("test", mode, float(ds.e[rows["test"]].mean()), 1000 + d)
            tau_d = tune_thresholds(ov_["P"], ov_["y"], FINE_GRID)
            pred = ot_["P"] >= tau_d
            per.append(scores(ot_["P"], ot_["u"], pred, y, ev, subsets))
            stack.append(pred)
        res[name] = {**mean_dicts(per), "draws": draws}
        preds[f"{name}_pred"] = np.stack(stack)
    for name in ("P5_replaced", "P5_half"):
        if name not in protocols:
            continue
        kind_c, draws = CORRUPTIONS[name]
        per, stack = [], []
        for d in range(draws):
            o = run("test", data=corrupt(ds, rows["test"], kind_c, 2000 + d))
            r, pred = corrupted(run, o, tau, ref, ev, subsets)
            per.append(r)
            stack.append(pred)
        res[name] = {**mean_dicts(per), "draws": draws}
        preds[f"{name}_pred"] = np.stack(stack)
    if "P5_gate" in protocols:
        tau_gate = gate_threshold(ds, rows["val"], ec["gate_max_reject"])
        clean = run("test", data=gated(Override(ds), rows["test"], tau_gate))
        per, stack = [], []
        for d in range(CORRUPTIONS["P5_replaced"][1]):
            o = run("test", data=gated(corrupt(ds, rows["test"], "lift", 2000 + d), rows["test"], tau_gate))
            r, pred = corrupted(run, o, tau, ref, ev, subsets)
            per.append(r)
            stack.append(pred)
        res["P5_gate"] = {**mean_dicts(per), "tau_gate": tau_gate,
                          "clean_macro_f1": scores(clean["P"], clean["u"], clean["P"] >= tau, y, ev, subsets)["all"][
                              "macro_f1"]}
        preds["P5_gate_pred"] = np.stack(stack)
    if "P5prime" in protocols and ds.mode == "cached":
        manifest, mut = load_mutations(cache, feats_dir)
        if manifest is not None:
            test_set = set(rows["test"].tolist())
            manifest = [r for r in manifest if r["row"] in test_set]
            pos = {int(i): k for k, i in enumerate(rows["test"])}
            per, stack, shares = [], [], []
            for d in range(max(r["draw"] for r in manifest) + 1 if manifest else 0):
                ovr, valid_rows = mutated(ds, manifest, d, mut)
                valid = np.zeros(len(rows["test"]), bool)
                valid[[pos[i] for i in valid_rows if i in pos]] = True
                o = run("test", data=ovr)
                r, pred = corrupted(run, o, tau, ref, ev, subsets, valid)
                per.append(r)
                stack.append(pred)
                shares.append(valid.mean())
            if per:
                res["P5prime"] = {**mean_dicts(per), "valid_share": float(np.mean(shares)), "draws": len(per)}
                preds["P5prime_pred"] = np.stack(stack)
    if "P6" in protocols:
        res["P6"] = {}
        for kind_c in P6_KINDS:
            res["P6"][kind_c] = {}
            for rho in RHOS:
                o = run("test", data=broken_share(ds, rows["test"], kind_c, rho, 3000))
                res["P6"][kind_c][str(rho)] = scores(o["P"], o["u"], o["P"] >= tau, y, ev, {"all": subsets["all"]})[
                    "all"]["macro_f1"]
    return res, preds


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


def summarize(per_seed: dict, preds: dict, lm) -> dict:
    flat = {s: flatten(r) for s, r in per_seed.items()}
    keys = sorted(set().union(*[f.keys() for f in flat.values()]))
    summ = {}
    for k in keys:
        vals = np.asarray([f[k] for f in flat.values() if k in f and f[k] is not None], float)
        if len(vals):
            summ[k] = {"mean": float(nanmean(vals)), "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                       "n": int(len(vals))}
    any_seed = next(iter(preds.values()))
    y, kind = any_seed["y"], any_seed["kind"]
    ci = {}
    for proto in [k[:-5] for k in any_seed if k.endswith("_pred")]:
        sets = []
        for p in preds.values():
            x = p.get(f"{proto}_pred")
            if x is not None:
                sets.extend(list(x) if x.ndim == 3 else [x])
        for sub in SUBSETS:
            rows = np.ones(len(y), bool) if sub == "all" else kind == sub
            ci[f"{proto}.{sub}"] = bootstrap_ci([s[rows] for s in sets], y[rows], lm.eval_idx)
    return {"metrics": summ, "ci95": ci, "seeds": sorted(per_seed)}


def markdown(name: str, summ: dict) -> str:
    m = summ["metrics"]
    cell = lambda k: f"{m[k]['mean']:.3f} ± {m[k]['std']:.3f}" if k in m else "–"
    lines = [f"# {name} (seeds {summ['seeds']})", "", "| Protocol | All | SolidiFI | SmartBugs |", "|---|---|---|---|"]
    for proto in ALL:
        if f"{proto}.all.macro_f1" in m:
            lines.append(f"| {proto} | {cell(f'{proto}.all.macro_f1')} | {cell(f'{proto}.solidifi.macro_f1')} | "
                         f"{cell(f'{proto}.smartbugs.macro_f1')} |")
    lines += ["", "| P1 metric | Value |", "|---|---|"]
    for k in ("auprc", "ece", "aurc", "risk_at_80"):
        lines.append(f"| {k} | {cell(f'P1.all.{k}')} |")
    for k in ("u_max", "coverage", "selective_risk", "risk_full"):
        lines.append(f"| deferral {k} | {cell(f'P1.deferral.{k}')} |")
    return "\n".join(lines) + "\n"


def analyzer_rows(cfg: dict, lm) -> dict:
    p, _, ds = datasets(cfg, train_caps=False)
    test = np.flatnonzero(p.meta.split.to_numpy() == "gold_test")
    kind = p.meta.kind.to_numpy()[test]
    y = ds.y_gold[test]
    preds = {a: ds.y_an[test, i] * ds.obs_an[test, i] for i, a in enumerate(ANALYZERS)}
    preds["union"] = np.maximum(*preds.values())
    out = {}
    for name, pred in preds.items():
        out[name] = {sub: float(nanmean(f1_per_class(pred[rows], y[rows])[list(lm.eval_idx)]))
                     for sub, rows in (("all", np.ones(len(kind), bool)), ("solidifi", kind == "solidifi"),
                                       ("smartbugs", kind == "smartbugs"))}
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate every seed of a run on the gold test set")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--protocols", nargs="+", default=list(ALL))
    ap.add_argument("--analyzers", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    lm = load_label_map()
    root = Path(cfg["run"]["eval"])
    if args.analyzers:
        out = root / "analyzers"
        out.mkdir(parents=True, exist_ok=True)
        rows = analyzer_rows(cfg, lm)
        json.dump(rows, open(out / "summary.json", "w"), indent=1)
        print(json.dumps(rows, indent=1))
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = {s: Path(cfg["run"]["ckpt"]) / run_name(cfg, s) / "best.pt" for s in cfg["seeds"]}
    ckpts = {s: p for s, p in ckpts.items() if p.exists()}
    if not ckpts:
        raise SystemExit(f"no checkpoint for {cfg['run']['name']} under {cfg['run']['ckpt']}")
    out = root / cfg["run"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    per_seed, preds, data = {}, {}, None
    for seed, path in sorted(ckpts.items()):
        model, ck = load_checkpoint(path, dev, lm.K)
        if data is None:
            p, _, ds = datasets(ck["cfg"], train_caps=False)
            split = p.meta.split.to_numpy()
            rows = {"val": np.flatnonzero(split == "gold_val"), "test": np.flatnonzero(split == "gold_test")}
            bad = np.isnan(ds.e) | np.isnan(ds.w)
            ds.e[bad], ds.w[bad] = ck.get("e_bar", 0.5), 1.0
            cache = Path(ck["cfg"]["data"]["cache"])
            data = (ds, rows, cache, cache / "feats" / ck["cfg"]["data"]["feats"])
        ds, rows, cache, feats_dir = data
        bf16 = ck["cfg"]["train"]["bf16"] and dev.type == "cuda"
        run = Runner(model, ds, rows, dev, bf16, cfg["train"]["eval_batch"], args.workers)
        res, pr = evaluate_seed(run, args.protocols, cfg["eval"], lm, cache, feats_dir)
        per_seed[seed], preds[seed] = res, pr
        json.dump(clean(res), open(out / f"s{seed}.json", "w"), indent=1)
        np.savez_compressed(out / f"pred_s{seed}.npz", **pr)
        print(f"seed {seed}: P1 macro-F1 {res['P1']['all']['macro_f1']:.4f}")
    summ = summarize(per_seed, preds, lm)
    json.dump(clean(summ), open(out / "summary.json", "w"), indent=1)
    (out / "summary.md").write_text(markdown(cfg["run"]["name"], summ))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
