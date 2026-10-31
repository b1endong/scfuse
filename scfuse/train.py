import argparse
import csv
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from scfuse.config import load_config, run_name
from scfuse.data import (Collate, ContractDataset, ContractSampler, CorruptTrain, availability, load_packed)
from scfuse.labels import ANALYZERS, load_label_map
from scfuse.metrics import f1_per_class, no_skill_f1, tune_thresholds
from scfuse.model.fusion import fuse
from scfuse.model.losses import (AnalyzerNoise, TransitionNoise, anchor_rates, caa_contract_loss, caa_function_terms,
                                  caa_span_terms, conflict_loss, evidential_loss, majority_loss, reliability_loss,
                                  transition_regularizers)
from scfuse.model.scfuse import SCFuse
from scfuse.schema import SOURCES, VI, VIEWS
from scfuse.utils.ast import LABELS

LOG_FIELDS = ["epoch", "step", "lr_head", "loss", "task", "weak", "caa", "fn", "span", "rel", "corrupt", "aux",
              "noise", "contracts_per_s", "val_macro_f1", "val_u", *[f"omega_{s}" for s in SOURCES], "seconds"]
PAIRS = {"SG": ("S", "G"), "SD": ("S", "D"), "GD": ("G", "D"), "AG": ("A", "G"), "AD": ("A", "D")}
VAL_KEYS = ("P", "u", "omega", "m_src")


def to_device(x, dev):
    if isinstance(x, dict):
        return {k: to_device(v, dev) for k, v in x.items()}
    return x.to(dev, non_blocking=True) if torch.is_tensor(x) else x


def versions() -> dict:
    import transformers
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "numpy": np.__version__, "git": git or None,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def stratified_subset(rows: np.ndarray, y: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng, frac = np.random.default_rng(seed), n / max(len(rows), 1)
    chosen: set[int] = set()
    for k in np.argsort(y[rows].sum(0), kind="stable"):
        pos = rows[y[rows, k] > 0]
        need = int(np.ceil(frac * len(pos))) - sum(int(y[r, k] > 0) for r in chosen)
        free = [r for r in pos if r not in chosen]
        if need > 0 and free:
            chosen.update(int(r) for r in rng.choice(free, min(need, len(free)), replace=False))
    rest = [r for r in rows if r not in chosen]
    if len(chosen) < n and rest:
        chosen.update(int(r) for r in rng.choice(rest, min(n - len(chosen), len(rest)), replace=False))
    return np.sort(np.fromiter(chosen, int))


def splits(meta, overfit: int | None, seed: int, dc: dict) -> dict:
    s, kind = meta.split.to_numpy(), meta.kind.to_numpy()
    calib = np.flatnonzero(s == "gold_anchor")
    if overfit:
        sub = np.sort(np.random.default_rng(seed).choice(calib, overfit, replace=False))
        return {"weak": np.zeros(0, int), "anchor": sub, "calib": sub, "val": sub}
    anchor = calib[np.isin(kind[calib], dc["anchor_kinds"])]
    if dc.get("anchor_n"):
        anchor = stratified_subset(anchor, np.stack(meta.y_gold.to_numpy()), dc["anchor_n"])
        calib = anchor
    weak = np.flatnonzero(s == "train")
    if dc["weak_subset"] == "mythril5k":
        weak = weak[np.stack(meta.obs_mythril.to_numpy())[weak].any(1)]
    elif dc["weak_subset"] == "none":
        weak = weak[:0]
    elif dc["weak_subset"] != "all":
        raise ValueError(dc["weak_subset"])
    return {"weak": weak, "anchor": anchor, "calib": calib, "val": np.flatnonzero(s == "gold_val")}


def fill_propensity(dss, idx: dict, cfg: dict) -> float:
    ds = dss[0]
    bad = np.isnan(ds.e) | np.isnan(ds.w)
    train = np.concatenate([idx["weak"], idx["anchor"]])
    if bad[train].any() and cfg["data"]["missing_propensity"] != "uniform":
        raise SystemExit(f"{bad[train].sum()} training contracts lack e(z)/w(z): run scfuse.prep.propensity, then pack")
    rows = idx["weak"] if len(idx["weak"]) else idx["anchor"]
    e_bar = cfg["mask"]["e_bar"]
    for d in dss:
        d.e[bad] = 0.5 if e_bar is None else e_bar
        d.w[bad] = 1.0
    return float(e_bar if e_bar is not None else ds.e[rows].mean())


def base_rate(ds, idx: dict, source: str) -> np.ndarray:
    if source == "anchor":
        y = ds.y_gold[idx["calib"]]
        return np.clip((y.sum(0) + 1) / (len(y) + 2), 0.01, 0.99)
    obs, y = ds.obs_an[idx["weak"]], ds.y_an[idx["weak"]]
    n = obs.max(1).sum(0)
    a = np.where(n > 0, (obs * y).max(1).sum(0) / np.maximum(n, 1), ds.y_gold[idx["calib"]].mean(0))
    return np.clip(a, 0.01, 0.99)


def audit_weights(batch: dict, lam_u: float) -> torch.Tensor:
    o = torch.maximum((batch["y_gold"] > 0).float(), batch["audited_all"].unsqueeze(1))
    return o + (1 - o) * lam_u


def batch_weights(w: torch.Tensor, clip: float) -> torch.Tensor:
    wc = w.clamp(max=clip)
    return len(w) * wc / wc.sum().clamp_min(1e-12)


def gold_loss(out, batch, model, lam_kl: float, theta, eta: float):
    y = batch["y_gold"]
    if model.head == "evidential":
        l = evidential_loss(out["R"], out["S"], y, model.a, model.W, lam_kl, theta)
        if eta:
            per = evidential_loss(out["r"], out["s"], y.unsqueeze(1), model.a, model.W, lam_kl, theta.unsqueeze(1))
            l = l + eta * (out["m_src"] * per).sum(1)
        return l
    bce = lambda z, t: (theta * F.binary_cross_entropy_with_logits(z.float(), t, reduction="none")).sum(-1)
    l = sum(bce(out[k], y) for k in ("logit", "teacher_logit") if k in out)
    if "view_logit" in out:
        per = torch.stack([bce(out["view_logit"][:, v], y) for v in range(out["view_logit"].shape[1])], 1)
        l = l + (out["view_mask"].float() * per).sum(1)
    return l


def tmnr_loss(out, batch, model, noise: TransitionNoise, cfg, lam_kl: float):
    g, y = batch["is_gold"].unsqueeze(1), batch["y_gold"]
    t = torch.where(g > 0, y, (batch["y_an"] * batch["obs_an"]).amax(1))
    keep = torch.where(g > 0, torch.ones_like(y), batch["obs_an"].amax(1))
    mv = out["m_src"][:, :noise.V]
    r2, s2, T = noise.transfer(batch["idx"], out["r"], out["s"])
    _, _, R2, S2 = fuse(r2, s2, torch.ones_like(mv), mv, model.a, model.W, model.rule)
    per = evidential_loss(R2, S2, t, model.a, model.W, lam_kl, keep)
    per = per + (mv * evidential_loss(r2, s2, t.unsqueeze(1), model.a, model.W, lam_kl, keep.unsqueeze(1))).sum(1)
    l_conf, l_cons = transition_regularizers(T, (r2 + s2) / (r2 + s2 + model.W), t, keep, mv)
    tc = cfg["supervision"]["tmnr"]
    return per, tc["lam_conf"] * l_conf + tc["lam_cons"] * l_cons


def late_fusion_regularizer(out, batch, model, cfg):
    g = batch["is_gold"].unsqueeze(1)
    t = torch.where(g > 0, batch["y_gold"], (batch["y_an"] * batch["obs_an"]).amax(1))
    keep = torch.where(g > 0, torch.ones_like(t), batch["obs_an"].amax(1)).unsqueeze(1)
    z, vm = out["view_logit"].float(), out["view_mask"].float()
    if "pdf_conf" in out:
        p_true = torch.where(t.unsqueeze(1) > 0, torch.sigmoid(z), 1 - torch.sigmoid(z)).detach()
        se = keep * (out["pdf_conf"].float() - p_true) ** 2 * vm.unsqueeze(-1)
        return se.sum() / (keep * vm.unsqueeze(-1)).sum().clamp_min(1)
    loss = (keep * F.binary_cross_entropy_with_logits(z, t.unsqueeze(1).expand_as(z), reduction="none")).sum(-1).detach()
    hist = model.qmf_hist
    hist[batch["idx"], :, 0] += loss * vm
    hist[batch["idx"], :, 1] += vm
    kap = hist[batch["idx"], :, 0] / hist[batch["idx"], :, 1].clamp_min(1)
    w, reg = out["qmf_conf"].float(), z.new_zeros(())
    for v in range(w.shape[1]):
        ok = vm[:, v] > 0
        wi, ki = w[ok, v], kap[ok, v]
        if len(wi) > 1:
            dw = wi[:, None] - wi[None, :]
            reg = reg + torch.relu(torch.sign(dw) * (ki[:, None] - ki[None, :]) + dw.abs()).mean()
    return cfg["model"]["baselines"]["qmf_lambda"] * reg


def caa(out, m, lc: dict, pairs) -> dict:
    if lc["caa"] == "none":
        return {}
    if lc["caa"] == "contract":
        return {"caa": caa_contract_loss(out["h"], out["mask"], m, pairs, lc["t"])}
    terms = caa_function_terms(out["h"], out["sel"], out["mask"], m, pairs, lc["t"])
    l_fn = torch.cat(terms).mean() if terms and sum(len(x) for x in terms) else m.new_zeros(())
    parts = {"fn": l_fn, "caa": l_fn}
    an = out.get("anchors")
    if lc["lam_sp"] and an is not None and lc["caa"] == "function+span":
        sp = out["sp_meta"]
        live = an["valid"] & (m[sp["b"], VI["S"]] > 0)
        st = caa_span_terms(an["block"], an["span"], sp["span"], sp["b"], sp["unit"], sp["lo"], sp["hi"], live, lc["t"])
        l_sp = torch.cat(st).mean() if st and sum(len(x) for x in st) else m.new_zeros(())
        parts.update(span=l_sp, caa=l_fn + lc["lam_sp"] * l_sp)
    return parts


def objective(out, batch, model, noise, cfg, lam_kl: float, an_view: list, n_weak: int):
    lc, sc = cfg["loss"], cfg["supervision"]
    g, y, m = batch["is_gold"], batch["y_gold"], out["m_src"]
    theta = audit_weights(batch, lc["lam_u"])
    wbar = batch_weights(batch["w"], cfg["train"]["w_clip"]) if not cfg["train"]["uniform_w"] else torch.ones_like(g)
    B, weak = len(g), 1 - g
    aux = out.get("aux")
    if sc["weak"] == "tmnr":
        l_task, aux = tmnr_loss(out, batch, model, noise, cfg, lam_kl)
        l_weak = None
        task = l_task
    else:
        task = g * gold_loss(out, batch, model, lam_kl, theta, lc["eta"])
        l_weak = None
        obs = batch["obs_an"]
        if sc["weak"] == "viewbound":
            obs = obs * m[:, an_view].unsqueeze(-1)
        if isinstance(noise, AnalyzerNoise) and n_weak:
            l_weak = noise(out["P"], batch["y_an"], obs, detach_rates=True)
        elif sc["weak"] == "majority" and n_weak:
            l_weak = majority_loss(out["P"], batch["y_an"], obs)
            if "teacher_logit" in out:
                l_weak = l_weak + majority_loss(torch.sigmoid(out["teacher_logit"].float()), batch["y_an"], obs)
            if "view_logit" in out:
                l_weak = l_weak + sum(out["view_mask"][:, v].float() * majority_loss(
                    torch.sigmoid(out["view_logit"][:, v]), batch["y_an"], obs) for v in range(out["view_logit"].shape[1]))
        if l_weak is not None:
            task = task + lc["lam_w"] * weak * l_weak
    parts = {"task": (wbar * task).sum() / B}
    if l_weak is not None:
        parts["weak"] = (wbar * weak * l_weak).sum() / B
    total = parts["task"]
    pairs = [(VI[a], VI[b]) for a, b in (PAIRS[p] for p in lc["caa_pairs"])]
    parts.update(caa(out, m[:, :len(VIEWS)], lc, pairs))
    if "caa" in parts:
        total = total + lc["lam_a"] * parts["caa"]
    if model.head == "evidential" and model.learned_rel and g.any():
        bce, cnt = reliability_loss(out["omega_logit"], m, out["Pj"], y, theta)
        parts["rel"] = (g * bce).sum() / (g * cnt).sum().clamp_min(1)
        total = total + lc["lam_r"] * parts["rel"]
    if lc["conflict"] and model.head == "evidential":
        aux = (aux if aux is not None else 0) + lc["conflict"] * conflict_loss(out["r"], out["s"], m, model.W).mean()
    if "corrupt_logit" in out and "corrupt_target" in batch:
        present = m[:, VI["D"]]
        raw = F.binary_cross_entropy_with_logits(out["corrupt_logit"].float(), batch["corrupt_target"], reduction="none")
        parts["corrupt"] = (raw * present).sum() / present.sum().clamp_min(1)
        total = total + lc["lam_corrupt"] * parts["corrupt"]
    if "qmf_conf" in out or "pdf_conf" in out:
        aux = (aux if aux is not None else 0) + late_fusion_regularizer(out, batch, model, cfg)
    if aux is not None:
        parts["aux"] = aux
        total = total + aux
    noise_loss = None
    if isinstance(noise, AnalyzerNoise):
        obs = batch["obs_an"] * (m[:, an_view].unsqueeze(-1) if sc["weak"] == "viewbound" else 1)
        data = (wbar * weak * noise(out["P"].detach(), batch["y_an"], obs, detach_rates=False)).sum() / B
        scale = 1.0 if sc["prior_scale"] == "batch" else 1.0 / max(n_weak, 1)
        noise_loss = lc["lam_w"] * data + scale * noise.prior_nll()
        parts["noise"] = noise_loss
    return total, noise_loss, parts


@torch.no_grad()
def predict(model, loader, dev, bf16: bool, mode: str = "full", e_bar: float = 0.5, seed: int = 0, keys=VAL_KEYS) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    acc = {k: [] for k in keys + ("y", "idx")}
    for batch in loader:
        m = availability(batch["m"], batch["e"], mode, gen, e_bar)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=bf16):
            out = model(to_device(batch, dev), m.to(dev))
        for k in keys:
            acc[k].append(out[k].float().cpu().numpy())
        acc["y"].append(batch["y_gold"].numpy())
        acc["idx"].append(batch["idx"].numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def trainable_state(model) -> dict:
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad and n.startswith("text.bb.")}
    return {k: v for k, v in model.state_dict().items() if k not in frozen}


def load_model_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("text.bb.") for k in missing), (missing[:5], unexpected[:5])


def load_checkpoint(path: Path, dev, K: int):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SCFuse(ck["cfg"], K, torch.zeros(K), ck["q_dims"], ck["ast_vocab"])
    load_model_state(model, ck["model"])
    return model.to(dev).eval(), ck


def datasets(cfg: dict, train_caps: bool = True):
    cache = Path(cfg["data"]["cache"])
    p = load_packed(cache, cfg["data"]["label_fix"])
    mode = "cached" if cfg["model"]["regime"] == "cached" else "tokens"
    feats = cache / "feats" / cfg["data"]["feats"]
    lc, ac = cfg["model"]["lora"], cfg["model"]["ast"]
    caps = (lc["max_units"], lc["max_len"]) if mode == "tokens" and train_caps else (None, None)
    ds_train = ContractDataset(p, mode, feats, *caps, ac["max_units"], ac["max_nodes"])
    ds_eval = ContractDataset(p, mode, feats, None, None, ac["max_units"], ac["max_nodes"])
    ds_eval.has_spans = False
    for v, names in (cfg["data"].get("q_drop") or {}).items():
        for n in names:
            for d in (ds_train, ds_eval):
                d.q[v][:, d.layout[v].index(n)] = 0.0
    if mode == "cached" and (lack := [v for v in ("S", "D") if v in cfg["views"]["use"] and v not in ds_train.feats]):
        raise SystemExit(f"no {feats}/{lack[0]}.npy for view {lack[0]}: run scfuse.prep.encode")
    return p, ds_train, ds_eval


def main():
    ap = argparse.ArgumentParser(description="Train SCFuse")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit", type=int)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed, tc, lc, sc = cfg["seed"], cfg["train"], cfg["loss"], cfg["supervision"]
    name = run_name(cfg) + (f"_overfit{args.overfit}" if args.overfit else "")
    run_dir, ckpt_dir = Path(cfg["run"]["out"]) / name, Path(cfg["run"]["ckpt"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = tc["bf16"] and dev.type == "cuda"
    lm = load_label_map()
    p, ds_train, ds_eval = datasets(cfg)
    if lc["caa"] != "function+span":
        ds_train.has_spans = False
    idx = splits(p.meta, args.overfit, seed, cfg["data"])
    e_bar = fill_propensity([ds_train, ds_eval], idx, cfg)
    if cfg["data"]["shuffle_labels"]:
        rng = np.random.default_rng(seed)
        for group in (idx["weak"], idx["calib"]):
            perm = rng.permutation(group)
            for a in (ds_train.y_gold, ds_train.y_an, ds_train.obs_an):
                a[group] = a[perm]
    mask_train = "full" if args.overfit else cfg["mask"]["train"]
    drop_d = 0.0 if args.overfit else cfg["mask"]["train_d"]
    a = base_rate(ds_train, idx, sc["base_rate"])
    model = SCFuse(cfg, lm.K, torch.from_numpy(a).float(), ds_train.q_dims, p.layout.get("ast_vocab", len(LABELS))).to(dev)
    if model.head == "qmf":
        model.qmf_hist = torch.zeros(len(ds_train), len(VIEWS), 2, device=dev)
    oc = cfg["model"]["opcode"]
    ckpt_mop = Path(cfg["data"]["cache"]) / "feats" / oc["ckpt"] if oc["ckpt"] else None
    if ckpt_mop and ckpt_mop.exists():
        model.opcode.load_state_dict(torch.load(ckpt_mop, map_location=dev, weights_only=True)["state"])
    elif oc["freeze"]:
        raise SystemExit("model.opcode.freeze needs a pretrained encoder: run scfuse.prep.mop")
    an_view = [VI[v] for v in lm.views.values()]
    noise = None
    if sc["weak"] in ("viewbound", "agnostic") and len(idx["weak"]):
        c = idx["calib"]
        if sc["fallback_rates"] is None:
            lack = [f"{a_} on {g_}" for i, a_ in enumerate(ANALYZERS) for g_ in ("weak", "calib")
                    if not ds_train.obs_an[idx[g_], i].any()]
            if lack:
                raise SystemExit(f"no analyzer labels for {', '.join(lack)}: run scfuse.prep.analyzers, then pack")
        theta = audit_weights({"y_gold": torch.from_numpy(ds_train.y_gold[c]),
                               "audited_all": torch.from_numpy(ds_train.audited_all[c])}, lc["lam_u"])
        alpha0, beta0, use = anchor_rates(torch.from_numpy(ds_train.y_gold[c]), theta, torch.from_numpy(ds_train.y_an[c]),
                                          torch.from_numpy(ds_train.obs_an[c]), sc["eps0"], sc["min_anchor_pos"],
                                          sc["fallback_rates"])
        noise = AnalyzerNoise(alpha0, beta0, use, sc["kappa"]).to(dev)
    elif sc["weak"] == "tmnr":
        if model.head != "evidential":
            raise SystemExit("supervision.weak=tmnr needs model.head=evidential")
        noise = TransitionNoise(np.concatenate([idx["weak"], idx["anchor"]]), len(ds_train), len(VIEWS), lm.K).to(dev)
    elif sc["weak"] not in ("viewbound", "agnostic", "majority", "none"):
        raise NotImplementedError(sc["weak"])

    sampler = ContractSampler(idx["weak"], idx["anchor"], tc["batch_size"], tc["n_gold"], seed)
    nw = cfg["data"]["num_workers"]
    rows = np.concatenate([idx["weak"], idx["anchor"]])
    train_set = CorruptTrain(ds_train, rows, tc["corrupt_d"], seed) if tc["corrupt_d"] > 0 else ds_train
    loader = torch.utils.data.DataLoader(train_set, batch_sampler=sampler, collate_fn=Collate(ds_train), num_workers=nw,
                                         pin_memory=dev.type == "cuda", persistent_workers=nw > 0)
    val_loader = torch.utils.data.DataLoader(ds_eval, sampler=idx["val"].tolist(), batch_size=tc["eval_batch"],
                                             collate_fn=Collate(ds_eval), num_workers=nw)
    named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
    opt = torch.optim.AdamW([{"params": [q for n, q in named if n.startswith("text.")], "lr": tc["lr_adapter"]},
                             {"params": [q for n, q in named if not n.startswith("text.")], "lr": tc["lr_head"]}],
                            weight_decay=tc["weight_decay"])
    noise_opt = (torch.optim.AdamW(noise.parameters(), lr=sc["lr_noise"], weight_decay=0.0)
                 if isinstance(noise, AnalyzerNoise) else None)
    sparse_opt = (torch.optim.SparseAdam(list(noise.parameters()), lr=sc["tmnr"]["lr"])
                  if isinstance(noise, TransitionNoise) else None)
    total_steps = tc["max_steps"] or math.ceil(len(sampler) / tc["accum"]) * tc["max_epochs"]
    warm = max(1, int(tc["warmup"] * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm)))))

    start, step, best, bad = 0, 0, -1.0, 0
    patience = None if args.overfit else tc["patience"]
    finished = lambda ep, st, bd: ((patience is not None and bd >= patience) or ep + 1 >= tc["max_epochs"]
                                   or bool(tc["max_steps"] and st >= tc["max_steps"]))
    if args.resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location=dev, weights_only=False)
        load_model_state(model, ck["model"])
        for obj, key in ((noise, "noise"), (opt, "opt"), (noise_opt, "noise_opt"), (sparse_opt, "sparse_opt"),
                         (sched, "sched")):
            if obj is not None and ck.get(key) is not None:
                obj.load_state_dict(ck[key])
        start, step, best, bad = ck["epoch"] + 1, ck["step"], ck["best"], ck["bad"]
        torch.set_rng_state(ck["rng"].cpu())
        if finished(ck["epoch"], step, bad):
            print(f"run already finished at epoch {ck['epoch']} (best {best:.4f})")
            return
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        json.dump(versions(), open(run_dir / "versions.json", "w"), indent=1)
        setup = {"n": {k: int(len(v)) for k, v in idx.items()}, "e_bar": e_bar, "mask_train": mask_train,
                 "q_layout": p.layout["q"], "pack_version": p.layout.get("version"),
                 "base_rate": dict(zip(lm.codes, a.round(4).tolist())),
                 "val_no_skill_macro_f1": no_skill_f1(ds_eval.y_gold[idx["val"]], lm.eval_idx),
                 "params_trainable": sum(q.numel() for _, q in named)}
        if isinstance(noise, AnalyzerNoise):
            setup["noise_init"] = {an: {c: [round(float(noise.alpha0[i, k]), 3), round(float(noise.beta0[i, k]), 3),
                                            bool(noise.use[i, k])] for k, c in enumerate(lm.codes)}
                                   for i, an in enumerate(ANALYZERS)}
        json.dump(setup, open(run_dir / "setup.json", "w"), indent=1)
    log_path = run_dir / "log.csv"
    new_log = not log_path.exists() or start == 0
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.DictWriter(log_f, LOG_FIELDS, extrasaction="ignore")
    if new_log:
        log.writeheader()
    gen = torch.Generator()
    for epoch in range(start, tc["max_epochs"]):
        model.train()
        sampler.set_epoch(epoch)
        gen.manual_seed(seed * 1000 + epoch)
        lam_kl = min(1.0, epoch / max(lc["kl_anneal_epochs"], 1e-9))
        sums, n_seen, t0, it = {}, 0, time.time(), 0
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(loader):
            m = availability(batch["m"], batch["e"], mask_train, gen, e_bar, drop_d)
            b = to_device(batch, dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=bf16):
                out = model(b, m.to(dev))
            if "SP" in b:
                out["sp_meta"] = b["SP"]
            loss, noise_loss, parts = objective(out, b, model, noise, cfg, lam_kl, an_view, len(idx["weak"]))
            if noise_opt is not None:
                noise_opt.zero_grad(set_to_none=True)
                noise_loss.backward()
                noise_opt.step()
            (loss / tc["accum"]).backward()
            for k, v in {"loss": loss, **parts}.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n_seen += len(batch["idx"])
            if (it + 1) % tc["accum"] == 0 or it + 1 == len(sampler):
                torch.nn.utils.clip_grad_norm_([q for g_ in opt.param_groups for q in g_["params"]], tc["clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if sparse_opt is not None:
                    sparse_opt.step()
                    sparse_opt.zero_grad(set_to_none=True)
                step += 1
                if tc["max_steps"] and step >= tc["max_steps"]:
                    break
        train_s = time.time() - t0
        pv = predict(model, val_loader, dev, bf16)
        tau = tune_thresholds(pv["P"], pv["y"])
        f1 = float(np.nanmean(f1_per_class(pv["P"] >= tau, pv["y"])[list(lm.eval_idx)]))
        om = (pv["omega"] * pv["m_src"]).sum(0) / np.maximum(pv["m_src"].sum(0), 1)
        row = {"epoch": epoch, "step": step, "lr_head": opt.param_groups[1]["lr"],
               **{k: round(v / (it + 1), 5) for k, v in sums.items()}, "contracts_per_s": round(n_seen / train_s, 1),
               "val_macro_f1": round(f1, 4), "val_u": round(float(pv["u"].mean()), 4),
               **{f"omega_{s}": round(float(o), 4) for s, o in zip(SOURCES, om)}, "seconds": round(time.time() - t0, 1)}
        log.writerow(row)
        log_f.flush()
        print(json.dumps(row))
        if f1 > best:
            best, bad = f1, 0
            torch.save({"model": trainable_state(model), "noise": noise.state_dict() if noise is not None else None,
                        "epoch": epoch, "tau": tau, "cfg": cfg, "q_dims": ds_train.q_dims,
                        "ast_vocab": p.layout.get("ast_vocab", len(LABELS)), "e_bar": e_bar}, ckpt_dir / "best.pt")
            info = {"epoch": epoch, "macro_f1": f1, "tau": dict(zip(lm.codes, tau.round(3).tolist())),
                    "f1": {c: None if np.isnan(x) else round(float(x), 4)
                           for c, x in zip(lm.codes, f1_per_class(pv["P"] >= tau, pv["y"]))}}
            if isinstance(noise, AnalyzerNoise):
                al, be = noise.rates()
                info["noise"] = {an: {c: [round(float(al[i, k]), 4), round(float(be[i, k]), 4), bool(noise.use[i, k])]
                                      for k, c in enumerate(lm.codes)} for i, an in enumerate(ANALYZERS)}
            json.dump(info, open(run_dir / "val_best.json", "w"), indent=1)
        else:
            bad += 1
        torch.save({"model": trainable_state(model), "noise": noise.state_dict() if noise is not None else None,
                    "opt": opt.state_dict(), "noise_opt": noise_opt.state_dict() if noise_opt else None,
                    "sparse_opt": sparse_opt.state_dict() if sparse_opt else None, "sched": sched.state_dict(),
                    "epoch": epoch, "step": step, "best": best, "bad": bad, "rng": torch.get_rng_state()},
                   ckpt_dir / "last.pt")
        if finished(epoch, step, bad):
            break
    log_f.close()
    print(f"best val macro-F1 {best:.4f} -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
