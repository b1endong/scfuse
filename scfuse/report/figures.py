import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scfuse.labels import ANALYZERS, load_label_map
from scfuse.protocols import P6_KINDS, RHOS
from scfuse.report.results import Results

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
P6_RUNS = [("scfuse", "SCFuse"), ("prompt", "Missing-aware prompts"), ("sv_D", "Single view D"), ("tmc", "TMC"),
           ("tmnr", "TMNR"), ("shaspec", "ShaSpec"), ("prompt_dmask", "Prompts, D masked in training")]


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 1.5, "lines.markersize": 4.5,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "figure.dpi": 150})


def caption(ax, text: str):
    ax.set_title(text, loc="left", pad=6, color=INK)


def quartile_box(ax, x: float, q: list, color: str, width: float = 0.3, label=None):
    p5, p25, p50, p75, p95 = q
    ax.add_patch(plt.Rectangle((x - width / 2, p25), width, max(p75 - p25, 1e-4), facecolor=color, alpha=0.55,
                               edgecolor="white", linewidth=1, label=label))
    ax.plot([x - width / 2, x + width / 2], [p50, p50], color=INK, linewidth=1.2)
    ax.plot([x, x], [p5, p25], color=INK2, linewidth=0.8)
    ax.plot([x, x], [p75, p95], color=INK2, linewidth=0.8)


def fig_traits(R: Results, out, ref_run: str = "scfuse") -> bool:
    t = R.json("data", "traits.json")
    if t is None:
        return False
    lm = load_label_map()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axes[0, 0]
    names = {"weak_train": "Weak train", "weak_test": "Weak test", "gold_solidifi": "Gold test, SolidiFI",
             "gold_smartbugs": "Gold test, SmartBugs"}
    for c, (k, label) in enumerate(names.items()):
        v = np.asarray(t["recovery_ecdf"].get(k, []))
        if len(v):
            ax.step(v, np.linspace(0, 1, len(v)), where="post", color=SERIES[c],
                    label=f"{label} ({t['recovery_n'][k]:,})")
    sel = t["selector"]
    ax.text(0.03, 0.97, f"precision: S {sel['S']['precision']:.3f}, D {sel['D']['precision']:.3f}\n"
                        f"recall: S {sel['S']['recall']:.3f}, D {sel['D']['recall']:.3f}", transform=ax.transAxes,
            va="top", fontsize=7, color=INK2)
    ax.set(xlabel="Share of dispatcher selectors recovered in D", ylabel="Share of contracts (ECDF)", xlim=(0, 1),
           ylim=(0, 1.02))
    ax.legend(loc="center left", frameon=False)
    caption(ax, "(a) Selector recovery of the lifted view D")
    ax = axes[0, 1]
    cal = (t.get("propensity") or {}).get("e_calibration", [])
    if cal:
        n = np.asarray([c["n"] for c in cal], float)
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK2, linewidth=0.8)
        ax.plot([c["p"] for c in cal], [c["rate"] for c in cal], color=SERIES[0], zorder=2)
        ax.scatter([c["p"] for c in cal], [c["rate"] for c in cal], s=20 + 120 * n / n.max(), color=SERIES[0],
                   edgecolor="white", linewidth=1, zorder=3)
        ax.text(0.03, 0.97, f"Out-of-fold AUROC {t['propensity']['e_auroc_oof']:.3f}", transform=ax.transAxes,
                va="top", fontsize=7, color=INK2)
    ax.set(xlabel="Predicted propensity e(z)", ylabel="Share with verified source", xlim=(0, 1), ylim=(0, 1))
    caption(ax, "(b) Calibration of the propensity e(z)")
    ax = axes[1, 0]
    learned = {}
    for d in R.run_dirs(ref_run):
        vb = d / "val_best.json"
        if vb.exists() and "noise" in (r := json.load(open(vb))):
            for a, per in r["noise"].items():
                for c, (al, be, use) in per.items():
                    if use:
                        learned.setdefault((a, c), []).append((al, be))
    rates = t["analyzers"]["anchor_rates"]
    for c_i, a in enumerate(ANALYZERS):
        color, mk = SERIES[c_i], MARKERS[c_i]
        for c, (al, be, n1, n0) in rates[a].items():
            if n1 == 0 or n0 == 0:
                continue
            ax.scatter(be, al, color=color, marker=mk, s=24, edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(c, (be, al), textcoords="offset points", xytext=(4, 3), fontsize=6, color=INK2)
            if (a, c) in learned:
                la, lb = np.mean(learned[(a, c)], 0)
                ax.scatter(lb, la, facecolor="none", edgecolor=color, marker=mk, s=24, linewidth=1, zorder=3)
                ax.plot([be, lb], [al, la], color=color, linewidth=0.6, alpha=0.7)
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK2, linewidth=0.8)
    ax.set(xlabel="False-positive rate β", ylabel="True-positive rate α", xlim=(0, 0.75), ylim=(0, 1.05))
    ax.legend(handles=[Line2D([], [], color=SERIES[i], marker=MARKERS[i], linestyle="", label=a.capitalize())
                       for i, a in enumerate(ANALYZERS)]
                      + [Line2D([], [], color=INK2, marker="o", linestyle="", label="filled: anchor estimate"),
                         Line2D([], [], color=INK2, marker="o", markerfacecolor="none", linestyle="",
                                label="hollow: learned (seed mean)")], loc="lower right", frameon=False)
    caption(ax, "(c) Analyzer error rates on the gold anchor")
    ax = axes[1, 1]
    pr, kap = t["analyzers"]["weak_positive_rate"], t["analyzers"]["kappa"]
    x = np.arange(lm.K)
    for i, a in enumerate(ANALYZERS):
        v = [pr[a][c] if pr[a][c] else np.nan for c in lm.codes]
        ax.scatter(x + (i - 0.5) * 0.18, v, color=SERIES[i], marker=MARKERS[i], s=26, edgecolor="white",
                   linewidth=0.8, label=a.capitalize(), zorder=3)
    for k, c in enumerate(lm.codes):
        vs = [pr[a][c] for a in ANALYZERS if pr[a][c]]
        if len(vs) == 2:
            ax.plot([k - 0.09, k + 0.09], vs, color=GRID, linewidth=2, zorder=2)
        if kap.get(c) is not None:
            ax.annotate(f"{kap[c]:.2f}", (k, max(vs) if vs else 1e-3), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(x, lm.codes)
    ax.set(ylabel="Weak contracts labeled positive")
    ax.legend(loc="lower left", frameon=False)
    caption(ax, f"(d) Analyzer positive rates on {t['analyzers']['mythril_subset']:,} weak contracts (labels: κ)")
    fig.tight_layout()
    fig.savefig(out / "fig_traits.pdf")
    plt.close(fig)
    return True


def fig_uncertainty(R: Results, out, run: str = "scfuse") -> bool:
    seeds, preds = R.seed_results(run), R.preds(run)
    if not seeds:
        return False
    lm = load_label_map()
    ev = list(lm.eval_idx)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    curves = [np.asarray(r["P1"]["risk_curve"]) for r in seeds.values()]
    cov = np.arange(1, len(curves[0]) + 1) / len(curves[0])
    a1.fill_between(cov, np.min(curves, 0), np.max(curves, 0), color=SERIES[0], alpha=0.18, linewidth=0)
    a1.plot(cov, np.mean(curves, 0), color=SERIES[0], label="SCFuse, ranked by u")
    if preds:
        err = np.mean([(p["P1_pred"][:, ev] != p["y"][:, ev].astype(bool)).mean(1) for p in preds.values()], 0)
        oracle = np.cumsum(np.sort(err)) / np.arange(1, len(err) + 1)
        a1.plot(cov, oracle, linestyle=":", color=INK2, linewidth=1, label="Oracle ranking")
        a1.axhline(err.mean(), linestyle="--", color=INK2, linewidth=0.8, label="Random ranking")
    aurc = R.metric(run, "P1.all.aurc")
    if aurc:
        a1.text(0.97, 0.05, f"AURC {aurc[0]:.3f}", transform=a1.transAxes, ha="right", fontsize=7, color=INK2)
    a1.set(xlabel="Coverage (share of contracts kept)", ylabel="Risk of the kept contracts", xlim=(0, 1))
    a1.legend(loc="upper left", frameon=False)
    caption(a1, "(a) Risk-coverage")
    q = np.asarray([r["P1"]["quintile_error"] for r in seeds.values()])
    xs = np.arange(1, q.shape[1] + 1)
    for row in q:
        a2.plot(xs, row, color=SERIES[0], alpha=0.35, linewidth=0.8)
    a2.plot(xs, q.mean(0), color=SERIES[0], marker="o", label=f"Mean of {len(q)} seeds")
    a2.set_xticks(xs, ["Q1\nlowest u", "Q2", "Q3", "Q4", "Q5\nhighest u"][:len(xs)])
    a2.set(ylabel="Error rate", ylim=(0, None))
    a2.legend(loc="upper left", frameon=False)
    caption(a2, "(b) Error rate per quintile of the fused uncertainty u")
    fig.tight_layout()
    fig.savefig(out / "fig_uncertainty.pdf")
    plt.close(fig)
    return True


def fig_lift(R: Results, out, run: str = "scfuse") -> bool:
    if R.summary(run) is None:
        return False
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1, 1, 1]})
    ax = axes[0]
    seeds = R.seed_results(run)
    for x, (proto, label) in enumerate((("P1", "Clean"), ("P5_replaced", "Replaced"), ("P5_half", "Half\ndropped"))):
        diags = [r[proto]["diag"] for r in seeds.values() if proto in r and "diag" in r[proto]]
        if diags:
            quartile_box(ax, x - 0.17, np.mean([d["bound_q"] for d in diags], 0), SERIES[0], 0.3,
                         "Bound" if x == 0 else None)
            quartile_box(ax, x + 0.17, np.mean([d["realized_q"] for d in diags], 0), SERIES[1], 0.3,
                         "Realized" if x == 0 else None)
    within = R.metric(run, "P1.diag.within_bound")
    if within:
        ax.text(0.03, 0.97, f"{100 * within[0]:.1f}% within the bound", transform=ax.transAxes, va="top", fontsize=7,
                color=INK2)
    ax.set_xticks(range(3), ["Clean", "Replaced", "Half\ndropped"])
    ax.set(xlim=(-0.5, 2.5), ylim=(0, 1), ylabel="Change of P without D, J")
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1, 0.88))
    caption(ax, "(a) Bound of Prop. 3")
    titles = {"absent": "(b) D absent", "lift": "(c) D replaced", "half": "(d) Half of D dropped"}
    handles = []
    for ax, kind in zip(axes[1:], P6_KINDS):
        for i, (r, label) in enumerate(P6_RUNS):
            vals = [R.metric(r, f"P6.{kind}.{rho}") for rho in RHOS]
            if all(v is None for v in vals):
                continue
            y = [np.nan if v is None else v[0] for v in vals]
            x = 100 * np.asarray(RHOS)
            if r == "scfuse":
                per = [[s["P6"][kind][str(rho)] for rho in RHOS] for s in R.seed_results(r).values() if "P6" in s]
                if len(per) > 1:
                    ax.fill_between(x, np.min(per, 0), np.max(per, 0), color=SERIES[i], alpha=0.15, linewidth=0)
            line, = ax.plot(x, y, color=SERIES[i], marker=MARKERS[i], label=label)
            if kind == "absent":
                handles.append(line)
        ax.set(xlabel="Broken D, ρ (%)", ylim=(0, 0.85))
        caption(ax, titles[kind])
    axes[1].set_ylabel("Macro-F1")
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.12))
    fig.tight_layout()
    fig.savefig(out / "fig_lift.pdf")
    plt.close(fig)
    return True
