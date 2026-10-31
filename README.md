# SCFuse

Smart-contract vulnerability detection by compiler-anchored, reliability-discounted evidential fusion of four program
views: verified source **S** and its AST **A** (optional), the bytecode control-flow graph **G** (always) and
pseudo-source **D** lifted from bytecode by a code foundation model (always, noisy). 

```
configs/            default.yaml (paper Table 6), smoke.yaml, v25.yaml, scfuse_r.yaml, baseline.yaml,
                    ablations/, baselines/, weak/ (P7), label_map.yaml
scfuse/
  prep/             index, source (solc AST, S and A, span anchors), cfg, lift (Qwen), lifted (D, F_D, F_sel),
                    leakage, analyzers, propensity, pack, encode, mop, mutate (P5'), traits
  utils/            evm (disassembly, abstract-interpretation CFG, dispatcher, fidelity, covariates),
                    solidity (lexer, fallback scanner, label-leak renaming), ast (static AST vocabulary), solc
  model/            encoders, fusion (Eq. 18, 20), certify (Prop. 3), losses, baselines, scfuse (SCFuse, SCFuse-R)
  data.py           packed dataset, contract batches, availability masks, D corruptions
  train.py          Algorithm 1
  protocols.py      P2-P6 view transformations
  eval.py           gold test under P1-P6, diagnostics, certificates, deferral
  stats.py          paired bootstrap with Holm correction
  seeds.py          gate G2
  sanity.py         memorization checks (Table 5)
  llm.py            zero-shot LLM baseline
  external.py       P8 external audited cohort
  report/           LaTeX tables and PDF figures from results/
scripts/slurm/      train.sbatch (one seed or an array of five), job.sbatch (any module)
tests/              fusion properties (B1), losses, bytecode and Solidity utilities
```

## Installation

```bash
conda activate scfuse          # Python 3.12, torch 2.4.0 + CUDA 12.1
pip install -r requirements.txt
solc-select install all        # recompilation of sources and lifts
```

## Data preparation (laptop; GPU steps on SLURM)

```bash
C="--cache data/scfuse --raw .."
python -m scfuse.prep.index $C
python -m scfuse.prep.source $C
python -m scfuse.prep.cfg $C
sbatch --job-name=lift scripts/slurm/job.sbatch scfuse.prep.lift $C --revision <sha> --splits gold_anchor gold_val gold_test
python -m scfuse.prep.lifted $C --lifter auto
python -m scfuse.prep.leakage $C
python -m scfuse.prep.analyzers import $C && python -m scfuse.prep.analyzers gold $C && python -m scfuse.prep.analyzers weak $C
python -m scfuse.prep.propensity $C
python -m scfuse.prep.pack $C
sbatch --job-name=encode scripts/slurm/job.sbatch scfuse.prep.encode --cache data/scfuse
sbatch --job-name=mop scripts/slurm/job.sbatch scfuse.prep.mop --cache data/scfuse
sbatch --job-name=mutate scripts/slurm/job.sbatch scfuse.prep.mutate --cache data/scfuse
python -m scfuse.prep.traits --cache data/scfuse
python -m scfuse.sanity --cache data/scfuse
```

## Training, evaluation, report

```bash
pytest -q
python -m scfuse.train --config configs/smoke.yaml
python -m scfuse.train --config configs/smoke.yaml --overfit 64 train.max_epochs=60 train.lr_head=1.0e-3
sbatch --array=0-4%3 scripts/slurm/train.sbatch configs/default.yaml
sbatch --array=0-4%3 scripts/slurm/train.sbatch configs/baselines/prompt.yaml
python -m scfuse.seeds --config configs/default.yaml
python -m scfuse.eval --config configs/default.yaml
python -m scfuse.eval --config configs/baselines/prompt.yaml
python -m scfuse.eval --analyzers
python -m scfuse.stats --ref scfuse --protocol P1 prompt shaspec tmc tmnr qmf pdf sv_D
python -m scfuse.external prepare --cohort cohort.jsonl --lifted <dir> --out results/external/dappscan_v2
python -m scfuse.external evaluate --prepared results/external/dappscan_v2 --config configs/default.yaml
python -m scfuse.external report --prepared results/external/dappscan_v2 --runs scfuse prompt shaspec_noD
python -m scfuse.report --results results --out figures
```

Every run writes `results/runs/<name>_s<seed>/` (config, versions, setup, `log.csv`, `val_best.json`) and
`checkpoints/<name>_s<seed>/{best,last}.pt`; resubmitting resumes. `scfuse.eval` writes
`results/eval/<name>/{s<seed>.json, pred_s<seed>.npz, summary.json, summary.md}`. The report reads only these files;
a missing result is printed as `--`.
