import argparse
from pathlib import Path

from scfuse.config import load_config
from scfuse.data import load_packed
from scfuse.report import figures, tables
from scfuse.report.results import Results

EXTERNAL_RUNS = ["sv_G", "sv_D", "vulnsense", "cmml", "shaspec", "shaspec_noD", "prompt", "tmc", "ecml", "tmnr", "qmf",
                 "pdf", "scfuse"]


def document(tabs: list[str], figs: list[str]) -> str:
    body = [f"\\input{{tables/{t}}}" for t in tabs]
    body += [f"\\begin{{figure*}}[t]\\centering\\includegraphics[width=\\textwidth]{{{f}}}\\end{{figure*}}" for f in figs]
    return "\n".join(["\\documentclass{article}", "\\usepackage[margin=2cm,landscape]{geometry}",
                      "\\usepackage{booktabs,graphicx,amsmath,amssymb}", "\\begin{document}", *body,
                      "\\end{document}", ""])


def main():
    ap = argparse.ArgumentParser(description="LaTeX tables and PDF figures from results/ (no value is typed by hand)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--cohort", default="dappscan_v2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    R, out = Results(args.results), Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    figures.style()
    tabs = {"tab_main": tables.tab_main(R), "tab_reliability": tables.tab_reliability(R),
            "tab_ablation": tables.tab_ablation(R), "tab_weak": tables.tab_weak(R),
            "tab_memorization": tables.tab_memorization(R), "tab_external": tables.tab_external(R, args.cohort,
                                                                                                EXTERNAL_RUNS)}
    if (Path(cfg["data"]["cache"]) / "packed" / "meta.parquet").exists():
        tabs["tab_data"] = tables.tab_data(load_packed(cfg["data"]["cache"], cfg["data"]["label_fix"]).meta)
    written = [name for name, tex in tabs.items() if tex]
    for name in written:
        (out / "tables" / f"{name}.tex").write_text(tabs[name])
    figs = [f"fig_{n}.pdf" for n, fn in (("traits", figures.fig_traits), ("uncertainty", figures.fig_uncertainty),
                                         ("lift", figures.fig_lift)) if fn(R, out)]
    (out / "benchmark.tex").write_text(document(written, figs))
    print(f"tables: {written}\nfigures: {figs}\n-> {out}")


if __name__ == "__main__":
    main()
