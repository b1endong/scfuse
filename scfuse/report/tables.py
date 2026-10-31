import numpy as np

from scfuse.labels import load_label_map
from scfuse.report.results import Results

REF = "scfuse"
METHODS = [
    ("analyzers:union", "Slither + Mythril"),
    ("llm_qwen2.5-coder-7b-instruct_S", "Zero-shot LLM, $\\mathsf{S}$"),
    ("llm_qwen2.5-coder-7b-instruct_D", "Zero-shot LLM, $\\mathsf{D}$"),
    None,
    ("sv_S", "Single view $\\mathsf{S}$ (UniXcoder)"),
    ("sv_S_codebert", "Single view $\\mathsf{S}$ (CodeBERT)"),
    ("sv_G", "Single view $\\mathsf{G}$"),
    ("sv_D", "Single view $\\mathsf{D}$"),
    ("sv_A", "Single view $\\mathsf{A}$ (AST)"),
    None,
    ("concat", "Concatenation"),
    ("attention", "Attention fusion"),
    ("vulnsense", "VulnSense-style"),
    ("cmml", "CMML-style"),
    ("shaspec", "ShaSpec"),
    ("prompt", "Missing-aware prompts"),
    ("prompt_dmask", "\\quad trained with $\\mathsf{D}$ masking"),
    None,
    ("tmc", "TMC"),
    ("ecml", "ECML"),
    ("tmnr", "TMNR"),
    ("qmf", "QMF"),
    ("pdf", "PDF"),
    None,
    ("scfuse", "SCFuse"),
    ("scfuse_r", "SCFuse-R"),
]
MAIN_COLS = [("P1.all.macro_f1", "All", "P1", 1), ("P1.solidifi.macro_f1", "SolidiFI", None, 1),
             ("P1.smartbugs.macro_f1", "SmartBugs", None, 1), ("P1.all.auprc", "AUPRC", None, 1),
             ("P1.all.ece", "ECE$\\downarrow$", None, -1), ("P1.all.aurc", "AURC$\\downarrow$", None, -1),
             ("P2.all.macro_f1", "P2", None, 1), ("P4.all.macro_f1", "P4", "P4", 1),
             ("P5_replaced.all.macro_f1", "Replaced", "P5_replaced", 1), ("P5_half.all.macro_f1", "Half", "P5_half", 1),
             ("P5_gate.all.macro_f1", "+gate", None, 1), ("P5prime.all.macro_f1", "P5$'$", None, 1)]
ABLATIONS = [
    ("Compiler-anchored alignment", [("abl_no_caa", "No CAA ($\\lambda_a=0$)"), ("abl_fn_only", "Function anchors only"),
                                     ("abl_contract_caa", "Contract-level CAA")]),
    ("Reliability-discounted fusion", [("abl_no_rel", "No learned reliability"),
                                       ("abl_no_fd", "No $F^{\\mathsf{D}}$ in $q^{\\mathsf{D}}$"),
                                       ("abl_ds", "Dempster--Shafer rule"), ("abl_no_J", "No joint source $\\mathsf{J}$"),
                                       ("abl_no_per_source", "No per-source loss")]),
    ("Views and training", [("abl_no_D", "No view $\\mathsf{D}$"), ("abl_no_S", "No view $\\mathsf{S}$"),
                            ("abl_no_A", "No view $\\mathsf{A}$"), ("abl_uniform_mask", "Uniform $\\mathsf{S}$ masking"),
                            ("abl_unit_w", "Unit weights $\\bar w\\equiv 1$"), ("abl_frozen", "Frozen encoders")]),
]
ABL_COLS = [("P1.all.macro_f1", "P1", "P1"), ("P4.all.macro_f1", "P4", "P4"),
            ("P5_replaced.all.macro_f1", "P5 repl.", "P5_replaced"), ("P5_half.all.macro_f1", "P5 half", "P5_half")]
WEAK = [("p7_viewbound", "View-bound (SCFuse)"), ("p7_agnostic", "View-agnostic"), ("p7_majority", "Majority vote"),
        ("p7_anchor_only", "Anchor-only")]


def f3(x) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pm(m) -> str:
    return "--" if m is None else f"{m[0]:.3f} $\\pm$ {m[1]:.3f}" if m[2] > 1 else f"{m[0]:.3f}"


def signed(x) -> str:
    return "--" if x is None else f"{x:+.3f}"


def table(colspec: str, head: list[str], body: list[str], caption: str, label: str, wide: bool = True) -> str:
    env = "table*" if wide else "table"
    return "\n".join([f"\\begin{{{env}}}[t]", "\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
                      f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *head, "\\midrule", *body, "\\bottomrule",
                      "\\end{tabular}", f"\\end{{{env}}}", ""])


def _cell(R: Results, run: str, key: str):
    if run.startswith("analyzers:"):
        s = R.json("eval", "analyzers", "summary.json")
        sub = key.split(".")[1]
        return (s[run.split(":")[1]][sub], 0.0, 1) if s and key.endswith("macro_f1") and key.startswith("P1") else None
    if run.startswith("llm_"):
        s = R.json("eval", run, "summary.json")
        sub = key.split(".")[1]
        return (s["macro_f1"][sub], 0.0, 1) if s and key.endswith("macro_f1") and key.startswith("P1") else None
    return R.metric(run, key)


def tab_main(R: Results) -> str:
    rows = [m for m in METHODS if m]
    vals = {(run, key): _cell(R, run, key) for run, _ in rows for key, *_ in MAIN_COLS}
    best = {}
    for key, _, _, sign in MAIN_COLS:
        xs = [(sign * v[0], run) for (run, k), v in vals.items() if k == key and v is not None]
        best[key] = max(xs)[1] if xs else None
    body = []
    for m in METHODS:
        if m is None:
            body.append("\\midrule")
            continue
        run, label = m
        n = R.n_seeds(run)
        cells = [label + ("$^{*}$" if n == 1 else "")]
        for key, _, proto, _ in MAIN_COLS:
            v = vals[(run, key)]
            s = f3(v[0] if v else None)
            if proto and run != REF and (p := R.p_holm(REF, proto, run)) is not None and p < 0.05:
                s += "$^{\\dagger}$"
            cells.append(f"\\textbf{{{s}}}" if best[key] == run and v is not None else s)
        body.append(" & ".join(cells) + " \\\\")
    head = ["& \\multicolumn{6}{c}{Full views (P1)} & \\multicolumn{2}{c}{No source} & "
            "\\multicolumn{4}{c}{Corrupted lifted view} \\\\",
            "\\cmidrule(lr){2-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-13}",
            "Method & " + " & ".join(h for _, h, _, _ in MAIN_COLS) + " \\\\"]
    cap = ("Macro-F1 on the gold test set (six classes), mean over seeds. $^{*}$one seed. $^{\\dagger}$paired difference "
           "from SCFuse with $p_{\\mathrm{Holm}}<0.05$. Bold: best per column. --: not applicable or not evaluated.")
    return table("l" + "c" * len(MAIN_COLS), head, body, cap, "tab:main")


def tab_reliability(R: Results) -> str:
    rows = [("P1.all.macro_f1", "Macro-F1, P1"), ("P5_replaced.all.macro_f1", "Macro-F1, $\\mathsf{D}$ replaced"),
            ("P5_half.all.macro_f1", "Macro-F1, half of $\\mathsf{D}$ dropped"), ("P5prime.all.macro_f1", "Macro-F1, P5$'$"),
            ("P5_replaced.diag.omega_D", "Mean $\\omega_{\\mathsf{D}}$, $\\mathsf{D}$ replaced"),
            ("P5_half.diag.omega_D", "Mean $\\omega_{\\mathsf{D}}$, half of $\\mathsf{D}$ dropped"),
            ("P5_replaced.diag.via_J", "$\\mathsf{D}$-affected evidence via $\\mathsf{J}$"),
            ("P5_replaced.diag.certified", "Certified decisions, $\\mathsf{D}$ replaced")]
    body = [label + " & " + " & ".join(pm(R.metric(run, key)) for run in ("scfuse", "scfuse_r")) + " \\\\"
            for key, label in rows]
    return table("lcc", ["& SCFuse & SCFuse-R \\\\"], body,
                 "Reliability diagnostics (mean $\\pm$ standard deviation over seeds).", "tab:reliability", False)


def tab_ablation(R: Results) -> str:
    ref = {key: R.metric(REF, key) for key, *_ in ABL_COLS}
    body = ["SCFuse (macro-F1) & " + " & ".join(f3(ref[k][0] if ref[k] else None) for k, *_ in ABL_COLS)
            + f" & {f3((R.metric(REF, 'P1.all.ece') or [None])[0])} \\\\"]
    for group, variants in ABLATIONS:
        body.append(f"\\multicolumn{{{len(ABL_COLS) + 2}}}{{l}}{{\\emph{{{group}}}}} \\\\")
        for run, label in variants:
            cells = [label]
            for key, _, proto in ABL_COLS:
                v = R.metric(run, key)
                s = signed(v[0] - ref[key][0] if v and ref[key] else None)
                if (p := R.p_holm(REF, proto, run)) is not None and p < 0.05:
                    s += "$^{\\dagger}$"
                cells.append(s)
            cells.append(f3((R.metric(run, "P1.all.ece") or [None])[0]))
            body.append(" & ".join(cells) + " \\\\")
    head = ["Variant & " + " & ".join(h for _, h, _ in ABL_COLS) + " & ECE \\\\"]
    return table("l" + "c" * (len(ABL_COLS) + 1), head, body,
                 "Ablation of SCFuse: difference in macro-F1 with the full model (seed means); ECE under P1. "
                 "$^{\\dagger}$: $p_{\\mathrm{Holm}}<0.05$.", "tab:ablation", False)


def tab_weak(R: Results) -> str:
    body = []
    for run, label in WEAK:
        cells = [label, pm(R.metric(f"{run}_56", "P1.all.macro_f1")), pm(R.metric(f"{run}_56", "P4.all.macro_f1")),
                 pm(R.metric(f"{run}_12", "P1.all.macro_f1"))]
        body.append(" & ".join(cells) + " \\\\")
    head = ["& \\multicolumn{2}{c}{56 anchors} & 12 anchors \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}",
            "Supervision & P1 & P4 & P1 \\\\"]
    return table("lccc", head, body, "Weak supervision (P7) on the doubly analyzer-labeled contracts: macro-F1, mean "
                                     "$\\pm$ standard deviation over seeds.", "tab:weak", False)


def tab_data(meta) -> str:
    lm = load_label_map()
    y = np.stack(meta.y_gold.to_numpy())
    ys = {a: np.stack(meta[f"y_{a}"].to_numpy()) * np.stack(meta[f"obs_{a}"].to_numpy()) for a in ("slither", "mythril")}
    obs = {a: np.stack(meta[f"obs_{a}"].to_numpy()).any(1) for a in ("slither", "mythril")}
    split, kind = meta.split.to_numpy(), meta.kind.to_numpy()
    body = []
    for name, sp in (("Anchor", "gold_anchor"), ("Validation", "gold_val"), ("Test", "gold_test")):
        for k in ("smartbugs", "solidifi"):
            r = (split == sp) & (kind == k)
            if r.any():
                body.append(f"{name} & {'SmartBugs' if k == 'smartbugs' else 'SolidiFI'} & gold & {r.sum():,} & "
                            + " & ".join(str(int(v)) for v in y[r].sum(0)) + " \\\\")
    for a in ("slither", "mythril"):
        r = (split == "train") & obs[a]
        cover = load_label_map().covered[a]
        body.append(f"Train & Ethereum & {a.capitalize()} & {r.sum():,} & "
                    + " & ".join(f"{int(v):,}" if c else "--" for v, c in zip(ys[a][r].sum(0), cover)) + " \\\\")
    body.append(f"Test & Ethereum & -- & {((split == 'test') & (kind == 'weak')).sum():,} & "
                + " & ".join("--" for _ in lm.codes) + " \\\\")
    head = ["Set & Source & Labels & Contracts & " + " & ".join(lm.codes) + " \\\\"]
    return table("lllr" + "r" * lm.K, head, body, "Benchmark composition: contracts and positive contracts per class.",
                 "tab:data")


def tab_memorization(R: Results) -> str:
    s = R.json("sanity", "memorization.json")
    if s is None:
        return ""
    rows = [("nearest_anchor.anchors_all", "Nearest anchor contract, anchors include SolidiFI"),
            ("nearest_anchor.anchors_smartbugs", "Nearest anchor contract, SmartBugs anchors only"),
            ("shared_function.anchors_all", "Function shared with a positive anchor, anchors include SolidiFI"),
            ("shared_function.anchors_smartbugs", "Function shared with a positive anchor, SmartBugs anchors only")]
    body = [label + " & " + " & ".join(f3(s[k][sub]) for sub in ("all", "solidifi", "smartbugs")) + " \\\\"
            for k, label in rows if k in s]
    return table("lccc", ["Check & All & SolidiFI & SmartBugs \\\\"], body,
                 "Memorization checks (macro-F1) on the gold test set.", "tab:memorization", False)


def tab_external(R: Results, cohort: str, runs: list[str]) -> str:
    rep = R.json("external", cohort, "report.json")
    if rep is None:
        return ""
    ci = lambda d: "--" if not d else f"{d['point']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"
    body = []
    for run in runs + ["runtime_size"]:
        r = rep.get(run)
        if not r:
            continue
        label = dict(m for m in METHODS if m).get(run, "Runtime size only" if run == "runtime_size" else run)
        body.append(" & ".join([label, str(r.get("seeds", "--")), ci(r.get("auroc")), f3(r.get("auroc_size_strata")),
                                f3(r.get("auprc")), ci(r.get("auroc_g_only")), ci(r.get("delta_auroc_vs_ref"))]) + " \\\\")
    head = ["Method & Seeds & AUROC & AUROC, size strata & AUPRC & AUROC, $\\mathsf{G}$ only & "
            "$\\Delta$AUROC vs SCFuse \\\\"]
    pos = ", ".join(f"{k} {v}" for k, v in rep["cohort"]["positives"].items())
    return table("lcccccc", head, body, f"External audit test (P8): {rep['cohort']['contracts']} contracts from "
                                        f"{rep['cohort']['projects']} projects (positives: {pos}); 95\\% project-"
                                        f"bootstrap intervals.", "tab:external")
