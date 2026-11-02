# Reproducing the experiments

This guide covers every step from the raw inputs to the tables and figures of the paper. All commands run from the
repository root. Steps marked *(GPU)* need a CUDA device; the others run on CPU. Each step reads and writes the cache
directory (`--cache`, default `data/scfuse`) or `results/`, so any step can be rerun once its inputs exist.

1. [Inputs](#1-inputs)
2. [Data preparation](#2-data-preparation)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)
5. [External audit test (P8)](#5-external-audit-test-p8)
6. [Tables and figures](#6-tables-and-figures)
7. [Sanity checks](#7-sanity-checks)
8. [Running on Slurm](#8-running-on-slurm)
9. [Environment variables](#9-environment-variables)

## 1. Inputs

The preparation scripts read two inputs. The frozen split files in `data/splits/` (`train.json`, `val.json`,
`test.json`, `gold_anchor.json`) list the weak contracts by address and the gold contracts by identifier, source path
and label. The raw benchmark artifacts lie under a root directory given with `--raw`:

```text
<raw>/data/
├── ethereum/contracts/mainnet/              Smart Contract Sanctuary mirror: <prefix>/<address>_<name>.sol, contracts.json
├── smartbugs_curated/, solidifi/            sources of the gold contracts (paths listed in data/splits/)
├── interim/contracts_with_bytecode.jsonl    runtime bytecode of the weak contracts
├── interim/dedup_clusters.json              clusters of exact and near-duplicate weak contracts
├── cache/gold_bytecode.json                 runtime bytecode of the gold contracts
├── processed/labels/weak_labels.jsonl       Slither results on the weak pool
├── processed/lifted/<id>.txt                Heimdall lifts (optional, see --lifter below)
├── raw/deployment_pool.parquet              deployment sample from Ethereum ETL: address, block_timestamp, bytecode
└── raw/deployment_population_etherscan.jsonl    verification status of deployment contracts (optional)
```

As stated in the paper, the split identifiers and the scripts that build these artifacts will be released upon
publication.

## 2. Data preparation

```bash
C="--cache data/scfuse --raw /path/to/raw"

python -m scfuse.prep.index $C                     # contract index from data/splits         -> index.jsonl
python -m scfuse.prep.source $C                    # views S and A: solc units, compiler match, block-span anchors -> units_S.jsonl
python -m scfuse.prep.cfg $C                       # view G: CFG, dispatcher selectors, q_G, covariates z -> cfg.jsonl
python -m scfuse.prep.lift $C --revision <commit>  # (GPU) view D: lifting with Qwen2.5-Coder-7B-Instruct -> lifted/
python -m scfuse.prep.lifted $C                    # D units, compilation, round-trip fidelity, selector agreement -> units_D.jsonl
python -m scfuse.prep.leakage $C                   # weak contracts that duplicate gold ones  -> excluded.json
python -m scfuse.prep.analyzers import $C          # Slither results of the weak pool         -> analyzers/
python -m scfuse.prep.analyzers gold $C            # Slither and Mythril on the gold contracts
python -m scfuse.prep.analyzers weak $C            # Mythril on 4,999 random weak training contracts
python -m scfuse.prep.propensity $C                # e(z) and w(z) from the deployment sample -> propensity.json
python -m scfuse.prep.pack $C                      # flat arrays of views, anchors, labels    -> packed/

python -m scfuse.prep.encode --cache data/scfuse   # (GPU) frozen UniXcoder features of S and D -> feats/unixcoder-base/
python -m scfuse.prep.mop --cache data/scfuse      # (GPU) masked-opcode pretraining of the block encoder -> feats/opcode_mop.pt
python -m scfuse.prep.mutate --cache data/scfuse   # (GPU) P5' mutations of the lifted view   -> mutations.json
python -m scfuse.prep.traits --cache data/scfuse   # low-quality traits (Fig. 3)              -> results/data/traits.json
python -m scfuse.sanity --cache data/scfuse        # memorization checks (Table 5)            -> results/sanity/memorization.json
```

- `prep.lift` needs a pinned Hugging Face commit of the lifter (`--revision`). It skips contracts already in its log,
  so an interrupted job resumes. `--shard i --num-shards n` splits the work across jobs, and `--splits` restricts it
  to some splits.
- `prep.lifted --lifter qwen` keeps only the foundation-model lifts. The default, `--lifter auto`, falls back to the
  Heimdall lift in `<raw>/data/processed/lifted` for a contract without one. `packed/layout.json` records how many
  contracts use each lifter.
- The analyzers time out after 60 s. Timeouts, crashes and classes an analyzer cannot report are stored as missing
  labels, never as negatives.
- Training stops with an error if `feats/opcode_mop.pt` is missing, since the opcode encoder is frozen after its
  pretraining.

**Smoke subset.** The same steps with `--cache data/smoke` and `scfuse.prep.index ... --weak-sample 200` build a small
subset for a quick pipeline check:

```bash
SCFUSE_SMOKE=data python -m scfuse.train --config configs/smoke.yaml
```

## 3. Training

```bash
python -m scfuse.train --config configs/default.yaml --seed 16            # one seed of SCFuse
python -m scfuse.train --config configs/default.yaml --seed 16 --resume   # continue an interrupted run
python -m scfuse.seeds --config configs/default.yaml                      # validation macro-F1 of every seed, mean and spread
```

Every configuration of the paper is a file in `configs/`. Table 7 marks the methods trained with a single seed; the
others use the five seeds listed in `configs/default.yaml` (16, 32, 64, 128, 256). Model selection and the class
thresholds use the gold validation set only; the thresholds are stored in `best.pt`. The default LoRA regime
fine-tunes adapters on the token ids in `packed/`, and the cached regime of the baselines and of the frozen-encoder
ablation reads the features written by `scfuse.prep.encode`.

Each run writes:

```text
results/runs/<name>_s<seed>/   config.yaml, versions.json, setup.json, log.csv, val_best.json
checkpoints/<name>_s<seed>/    best.pt (best validation macro-F1), last.pt (used by --resume)
```

## 4. Evaluation

```bash
python -m scfuse.eval --config configs/default.yaml                     # every seed with a checkpoint, all protocols
python -m scfuse.eval --config configs/default.yaml --protocols P1 P4   # a subset of the protocols
python -m scfuse.eval --analyzers                                       # Slither, Mythril and their union
python -m scfuse.llm --view S                                           # (GPU) zero-shot Qwen2.5-Coder-7B-Instruct on S
python -m scfuse.llm --view D                                           # (GPU) the same on D
python -m scfuse.stats --ref scfuse --protocol P1 prompt shaspec tmc tmnr qmf pdf sv_D
```

The protocol codes of `--protocols` are `P1`, `P2`, `P3`, `P4`, `P5_replaced`, `P5_half`, `P5_gate`, `P5prime` and
`P6` (definitions in the [README](../README.md#reproducing-the-paper)). P2 and P3 average ten mask draws and P5 five
corruption draws; P5′ needs the mutations written by `scfuse.prep.mutate`.

`scfuse.eval` writes `results/eval/<name>/` with `s<seed>.json` (metrics, diagnostics, certificates and deferral),
`pred_s<seed>.npz` (predictions), `summary.json` and `summary.md`. `scfuse.stats` compares runs with the reference by a
paired bootstrap over test contracts with Holm-adjusted p-values (`--subset solidifi|smartbugs` restricts it to one
source, `--groups host` resamples SolidiFI host contracts) and writes
`results/analysis/paired_<ref>_<protocol>_<subset>.json`.

## 5. External audit test (P8)

The cohort file is JSON Lines with one contract per line: `id`, runtime `bytecode`, `solc`, `project` and `labels`
(class code to 1, 0 or null for unknown). The lifts of the cohort are read from `<dir>/<id>.sol`.

```bash
python -m scfuse.external prepare --cohort cohort.jsonl --lifted <dir> --out results/external/dappscan_v2   # (GPU)
python -m scfuse.external evaluate --prepared results/external/dappscan_v2 --config configs/default.yaml
python -m scfuse.external report --prepared results/external/dappscan_v2 --runs scfuse prompt shaspec_noD
```

`evaluate` applies the checkpoints of a configuration, with G and D and with G only, and nothing is tuned on the
cohort. `report` computes macro AUROC, size-stratified AUROC and AUPRC with project-bootstrap intervals and writes
`results/external/dappscan_v2/report.json`.

## 6. Tables and figures

```bash
python -m scfuse.report --results results --out figures
```

The report reads only files under `results/`, prints a missing result as `--` and types no value by hand:

| Output | Paper |
|---|---|
| `figures/tables/tab_data.tex` | Table 4 |
| `figures/tables/tab_memorization.tex` | Table 5 |
| `figures/tables/tab_main.tex` | Table 7 |
| `figures/tables/tab_reliability.tex` | Table 8 |
| `figures/tables/tab_ablation.tex` | Table 9 |
| `figures/tables/tab_weak.tex` | Table 10 |
| `figures/tables/tab_external.tex` | Table 11 |
| `figures/fig_traits.pdf` | Figure 3 |
| `figures/fig_uncertainty.pdf` | Figure 5 |
| `figures/fig_lift.pdf` | Figure 6 |

`figures/benchmark.tex` collects all of them in one LaTeX document.

## 7. Sanity checks

```bash
pytest -q                                                                  # fusion properties, losses, bytecode and Solidity utilities
SCFUSE_SMOKE=data python -m scfuse.train --config configs/smoke.yaml --overfit 64 train.max_epochs=60 train.lr_head=1.0e-3
python -m scfuse.train --config configs/default.yaml data.shuffle_labels=true run.name=shuffled train.max_epochs=3
```

The overfitting run should fit its 64 contracts. With shuffled labels, the validation macro-F1 in `val_best.json`
should stay close to `val_no_skill_macro_f1` in `setup.json`.

## 8. Running on Slurm

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # once, on a login node
mkdir -p logs                                                                           # Slurm writes job logs here

sbatch scripts/slurm/train.sbatch configs/default.yaml 16                               # one seed
sbatch --array=0-4%3 scripts/slurm/train.sbatch configs/default.yaml                    # five seeds, at most three at a time
sbatch --job-name=encode scripts/slurm/job.sbatch scfuse.prep.encode --cache data/scfuse  # any module
```

Both scripts activate the virtual environment in `.venv` (`SCFUSE_ENV` points them to another one created with
`python -m venv`) and request one GPU, 8 CPU cores and 64 GB of memory. `train.sbatch` always passes `--resume`, so
resubmitting the same command continues unfinished runs and leaves finished ones unchanged.

## 9. Environment variables

| Variable | Default | Used for |
|---|---|---|
| `SCFUSE_DATA` | `data` | root of the packed data in the configurations (`$SCFUSE_DATA/scfuse`) |
| `SCFUSE_OUT` | `results` | run logs, evaluation results and analyses |
| `SCFUSE_CKPT` | `checkpoints` | checkpoints |
| `SCFUSE_CACHE` | `data/scfuse` | default `--cache` of the preparation scripts that take `--raw` |
| `SCFUSE_RAW` | `..` | default `--raw` of the same scripts |
| `SCFUSE_SMOKE` | `../SCFuse2.5/data` | root of the smoke data in `configs/smoke.yaml` |
| `SCFUSE_V25` | `../SCFuse2.5/data` | root of the release-2.5 data in `configs/v25.yaml` |
| `SCFUSE_ENV` | `.venv` | virtual environment activated by the Slurm scripts |
| `SOLC_ARTIFACTS` | `~/.solc-select/artifacts` | solc binaries installed by `solc-select` |
