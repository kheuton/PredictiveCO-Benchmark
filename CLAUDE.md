# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PredictiveCO-Benchmark** is a research codebase (NeurIPS 2024) for benchmarking Predict-then-Optimize (PtO) and Predict-and-Optimize (PnO) methods on combinatorial optimization problems. The package is named `openpto`.

## Installation

```bash
pip install -e .
```

Data must be placed under `./openpto/data/` (download from Google Drive link in README).

## Running Experiments

The main entrypoint is `rethink_exp/main_results.py`:

```bash
python rethink_exp/main_results.py \
  --problem knapsack \
  --opt_model mse \
  --solver gurobi \
  --n_epochs 300 \
  --gpu 0 \
  --prefix bench
```

**Key arguments** (see `openpto/config/utils_conf.py` for full list):
- `--problem`: `knapsack`, `portfolio`, `budgetalloc`, `energy`, `cubic`, `bipartitematching`, `advertising`, `shortestpath`, `TSP`
- `--opt_model`: `mse`, `dfl`, `blackbox`, `identity`, `spo`, `nce`, `qptl`, `pointLTR`, `pairLTR`, `listLTR`, `lodl`, `perturb`, `cpLayer`
- `--solver`: `gurobi`, `cvxpy`, `heuristic`, `neural`, `ortools`, `qptl`
- `--method_path`: path to model config YAML (default `openpto/config/models/default.yaml`)
- `--config_path`: path to problem config YAML (auto-detected from `--problem` if empty)
- `--prefix`: used to name output directories under `saved_records/`
- `--loadnew False`: reuse cached problem instance from `saved_problems/`

See `shells/benchmarks/` for per-problem example commands.

## Benchmark Fidelity (Critical)

**Always preserve these fixed settings** so results are directly comparable to the published NeurIPS 2024 table and to each other:

```
--seed 2023  --n_epochs 300  --patience 40  --pred_model dense  --n_ptr_epochs 0
```

Instance counts — **must not deviate**:

| Problem | `--instances` | `--testinstances` | Notes |
|---|---|---|---|
| knapsack | 400 (default) | 200 (default) | |
| knapsack-real | (don't pass) | (don't pass) | hardcoded real dataset |
| energy | (don't pass) | (don't pass) | hardcoded real dataset |
| budgetalloc | 400 (default) | 200 (default) | |
| cubic | **250** | **400** | non-default, must be explicit |
| bipartitematching | **20** | **6** | non-default, must be explicit |
| portfolio | 400 (default) | 200 (default) | |

Knapsack-real and energy silently ignore `--instances`/`--testinstances` (they load fixed real-world splits). Do not pass those flags for them.

**Solvers for comparability:**
- knapsack: `heuristic` (DP) — 100× faster than Gurobi, validated equivalent accuracy
- knapsack-real, energy: `gurobi`
- budgetalloc: `neural`
- cubic, bipartitematching: `heuristic` / `cvxpy`
- portfolio: `cvxpy`
- qptl, cpLayer: only valid for knapsack / bipartitematching / portfolio

## Hyperparameter Tuning Principle

The original benchmark only tuned learning rate. Our re-run sweeps **both LR and batch size**, plus method-specific HPs (dflalpha, lambd, tau, n_samples, sigma). This is the right way to compare methods fairly.

**Phase 1 (LR × Batch):** 3 LRs × 2 batch configs per method × task — establishes best training setup.
**Phase 2 (method HP):** sweeps the key HP for each method using Phase 1's best (LR, batch).

Always prefer results from the Phase 1/2 sweep over ad-hoc runs when reporting numbers.

## Running Tests

```bash
python -m pytest tests/test_perturbed_softdecision.py -v
```

## Benchmark Re-Run Sweep

The canonical benchmark comparison lives in the Phase 1/2 sweep infrastructure:

```bash
# Submit Phase 1 (498 jobs: all methods × all tasks × 3 LR × 2 batch)
bash shells/slurm/submit_bench_p1.sh --dry-run          # preview
bash shells/slurm/submit_bench_p1.sh                    # submit all
bash shells/slurm/submit_bench_p1.sh --problem knapsack # filter by problem
bash shells/slurm/submit_bench_p1.sh --method mse       # filter by method

# Monitor
python rethink_exp/sweep_status.py --phase 1            # status grid (✓/R/--)
python rethink_exp/sweep_status.py --phase 1 --vals     # best regret per cell

# Collect Phase 1 results → pick best (LR, batch) per method×task
python rethink_exp/collect_bench_p1.py                  # prints grid, writes bench_p1_best.json

# Submit Phase 2 (~210 jobs: method-specific HP sweep using Phase 1 best configs)
bash shells/slurm/submit_bench_p2.sh --dry-run
bash shells/slurm/submit_bench_p2.sh

# Collect Phase 2 → final table
python rethink_exp/collect_bench_p2.py --final
```

All jobs use `--requeue` + checkpoint/resume (SIGTERM handler in `ExpManager.py`). Jobs interrupted by preemption restart automatically from the last completed epoch.

**Prefix conventions:**
- Phase 1: `bench_p1_{method}_{batch}_lr{lr}` e.g. `bench_p1_mse_default_lr1e-2`
- Phase 2: `bench_p2_{method}_{hp_tag}_{batch}_lr{lr}` e.g. `bench_p2_dfl_alpha0.01_gd_lr5e-3`

## SLURM (Tufts HPC)

Submit experiments via scripts in `shells/slurm/`. The conda environment used is `pco_bench_rhel7`. Example:

```bash
bash shells/slurm/submit_softdec_wave1.sh --dry-run   # preview
bash shells/slurm/submit_softdec_wave1.sh              # submit
bash shells/slurm/submit_softdec_wave1.sh --problem knapsack
```

Each submit script supports `--dry-run`, `--problem <filter>`, and `--no-backup` flags. Partition: `hugheslab,batch`.

## Architecture

The pipeline has four modular components assembled in `rethink_exp/main_results.py`:

### 1. Problems (`openpto/problems/`)
All problems extend `PTOProblem` (abstract base in `PTOProblem.py`). Key methods:
- `get_train/val/test_data()` → `(X, Y, Y_aux)`
- `get_decision(Y, params, ptoSolver)` → `(Z, objectives)` — runs the CO solver
- `get_objective(Y, Z, aux_data)` → per-instance objective values
- `init_API()` → dict of solver init kwargs

Problems are cached to `saved_problems/` as pickles; use `--loadnew True` to force regeneration.

### 2. Solvers (`openpto/method/Solvers/`)
Selected by `solver_wrapper()` in `wrapper_solver.py`. Grouped by backend:
- `grb/`: Gurobi-backed solvers (knapsack, energy, advertising, TSP, portfolio-QP)
- `cvxpy/`: CvxpyLayer-backed differentiable solvers (portfolio, bipartite matching, knapsack)
- `heuristic/`: DP, shortest-path, TopK, LKH
- `neural/`: budget allocation, soft-TopK
- `ortools/`: advertising

### 3. Loss Functions / PnO Models (`openpto/method/Models/`)
All extend `optModel` (abstract base in `abcOptModel.py`), implementing `forward(problem, coeff_hat, coeff_true, params)` → `loss`. Registered in `wrapper_loss.py`:
- PtO: `MSE`, `BCE`, `CE`, `MAE`, `DFL`
- PnO: `SPO`, `QPTL`, `Blackbox`, `NCE`, `LTR` variants, `LODL`, `perturbed`, `cpLayer`

**`perturbed`** (in `perturbed.py`) implements the **correct Berthet et al. DPO** (soft-decision formulation). The legacy `perturbed_reinforce` class uses REINFORCE on scalar objectives (kept for reproducibility). The `perturbed` model supports a `SigmaScheduler` with schedules: `constant`, `linear_decay`, `cosine_decay`, `step_decay`. `ExpManager` calls `loss_fn.step(epoch)` each epoch if that method exists.

### 4. Prediction Models (`openpto/method/Predicts/`)
Neural nets mapping features X → predicted cost coefficients Ŷ. Selected via `--pred_model`: `dense`, `cvr`, `cv_mlp`, `ConvNet`, etc.

### Experiment Manager (`openpto/expmanager/ExpManager.py`)
Orchestrates the full train/eval loop:
1. Pretrain with two-stage (MSE) loss for `--n_ptr_epochs` epochs
2. Fine-tune with the chosen PnO loss for `--n_epochs` epochs (minibatch via DataLoader)
3. Evaluate on train/val/test; track best model by val regret; save checkpoints

**Checkpoint/resume support** (added for Phase 1/2 sweep):
- Saves `checkpoint_latest.pt` after every epoch (model, optimizer, best_val, time_since_best)
- Registers SIGTERM handler: saves checkpoint and exits 0 on preemption → SLURM `--requeue` restarts automatically
- On startup, detects `checkpoint_latest.pt` and resumes from the last completed epoch
- `train_logs.csv` / `val_logs.csv` written incrementally (append per epoch) so full history survives restarts

**Output directories:**
- `saved_records/<problem>-<version>/<opt_model>/<prefix>/` — latest run logs + checkpoints
- `saved_records/timed_logs/<problem>-<version>/<opt_model>/<prefix>/<timestamp>/` — timestamped backup

### Configs (`openpto/config/`)
- `config/probs/<problem>.yaml` — dataset params + solver kwargs
- `config/models/<name>.yaml` — hyperparams per opt_model (reduction, tau, n_samples, sigma, etc.)

Perturb model configs follow the naming convention `perturb_s<sigma>_n<n_samples>[_<noise>][_<activation>].yaml`.

## Visualization Code

**Don't** create Jupyter notebooks for visualization — each cell requires a separate tool call, bloating context unnecessarily.

**Do** write a plain `.py` script with section comments (`# ---- Section ----`) in place of markdown cells. Save figures with `plt.savefig()`. Convert to a notebook with `jupytext` only if interactive iteration is explicitly needed.

## Collecting Results

```bash
# Original benchmark scripts
python rethink_exp/collect_results.py    # prints benchmark table for all problems/models
python rethink_exp/find_best_val.py      # finds best hyperparams across prefix sweeps

# Phase 1/2 sweep scripts (preferred for the re-run)
python rethink_exp/collect_bench_p1.py  # Phase 1 grid + writes bench_p1_best.json
python rethink_exp/collect_bench_p2.py  # Phase 2 HP sweep results
python rethink_exp/collect_bench_p2.py --final   # full final table across all methods
```

Results for each run are stored in `saved_records/.../results.npy` as `[Objs_test_opt, eval_values]`.
