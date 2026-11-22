# certcorrect: certified post-solve correction of Galerkin contact ROMs

Code for the project described in `../FINAL_RESEARCH_PLAN.md` (mathematics: `../MATH_FORMULATION_v6c.md`,
`../THEOREM_certificate_efficiency.md`). Coding rules: `../Guidlines/SCRIPT_GUIDELINES.md`.

The package depends only on `fea_engine` and `rom_engine` (computation suite) and NumPy/SciPy/Matplotlib. It imports
no package of another research project. The L1 benchmarks re-implement the M2 definitions and give identical
matrices, checked once on 22 Nov 2026.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
```

Do not use the Anaconda base environment: importing `fea_engine` there aborts with an OpenMP conflict
(`libiomp5md.dll` loaded twice by MKL NumPy and torch).

## Layout

| Path | Content | Status |
|---|---|---|
| `certcorrect/problems.py` | membrane, block, layered block; affine `K(mu) = sum theta_q K_q` | done |
| `certcorrect/fom.py` | full-order solve (`solve_contact_static_sparse`), KKT residuals, energy errors | done |
| `certcorrect/rom.py` | Galerkin ROM G0 (`solve_reduced_contact`, affine offline projection) | done |
| `certcorrect/bases.py` | POD (Euclidean or energy), NMF / SVD dual bases | done |
| `certcorrect/certificate.py` | feasible-pair certificate, acceptance rule, min-theta | done |
| `certcorrect/corrections.py` | R0, R0s (selection fast path, thin-QR general case); B, Bs (weeks 5-6) | R0/R0s done |
| `certcorrect/active_set.py` | rules `multiplier`, `gap`; recall / over-snap; `neighbours` (weeks 5-6) | partly done |
| `certcorrect/multipliers.py` | S1 and S1-restricted (LCP via `rom_engine.solve_lcp`); S0, S2 (weeks 5-6) | S1 done |
| `certcorrect/pipeline.py`, `analysis.py` | snapshots, bases, `X = K(mu_bar)`, per-query evaluation; rates | done |
| `certcorrect/offline_online.py` | Riesz representers, Gram blocks (weeks 7-8) | interface |
| `certcorrect/verified.py` | Level 2 finite-precision bounds (week 13) | interface |
| `certcorrect/baselines/` | A (Zhang slack), G (enriched Galerkin), CS (conformal) (weeks 9-10) | interface |
| `certcorrect/config.py` | paths, seeds, presets, fixed acceptance thresholds | done |
| `certcorrect/results_io.py`, `plotting.py`, `cli.py`, `environment.py` | result files, figure style, CLI, run metadata | done |
| `experiments/x1_rate.py` | X1 rate test (`--rule multiplier|gap|both`, `--tau-lambda`, `--tau-gap`) | done |
| `experiments/x2_effectivity.py` ... `x8_finite_precision.py` | one script per experiment | stubs |
| `figures/fig_x1_rate.py` | X1 figure from saved results | done |
| `results/`, `paper/figures/`, `paper/tables/` | generated output only | |

## Running an experiment

```
.venv\Scripts\python experiments\x1_rate.py --problem membrane --seed 0 [--dry-run]
```

Exit codes: 0 ok, 1 failure, 2 bad input, 3 not implemented yet. Each result file
(`results/<experiment>/<problem>_seed<seed>.npz`) stores the arguments and the environment record (versions of
`fea_engine`, `rom_engine`, NumPy, SciPy, BLAS threads). Figures and tables are drawn from these files only.
