"""Paths, seeds, benchmark presets and fixed acceptance thresholds, in one place.

Thresholds are those fixed in FINAL_RESEARCH_PLAN.md (Secs. 6-7) before any experiment is run; change them only with a
recorded reason in the plan.
"""

from pathlib import Path

from .problems import ContactProblem, ElasticBlock, Membrane

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"
NUMBERS_TEX = ROOT / "paper" / "numbers.tex"

# Seeds of every random draw; experiment scripts take --seed and offset from these.
SEEDS = {
    "train": 0,          # training parameters (snapshots)
    "test": 1_000,       # in-range test queries
    "outside": 2_000,    # load-extrapolation and patch-shift queries (X5)
}

# Benchmarks (L1). Mesh sizes as in M2 so results are comparable; L3 refinement overrides n / n_xy.
PROBLEMS = {
    "membrane": lambda: Membrane(),
    "block": lambda: ElasticBlock(n_xy=12, nz=3),
    "layered": lambda: ElasticBlock(n_xy=16, nz=4, layered=True),
}

# Snapshot and basis settings per benchmark (basis sizes swept in X1).
SETTINGS = {
    # r_dual = 96 (not M2's 24): with 24 the G0 error plateaus at ~5 % from r = 40 on (week-2 sweep, 22 Nov 2026,
    # results/x1_rd*); with 96 it still decreases at r = 60. Runs with 24 are kept in results/x1*_rd24.
    "membrane": dict(n_train=200, n_test=200, r_primal=(5, 10, 20, 40, 60), r_dual=96, dual_kind="nmf"),
    "block": dict(n_train=100, n_test=100, r_primal=(4, 8, 15, 30), r_dual=12, dual_kind="nmf"),
    "layered": dict(n_train=100, n_test=100, r_primal=(4, 8, 15, 30), r_dual=12, dual_kind="nmf"),
}

# X2 mesh refinement (L3): constructors per level, and snapshot / basis sizes held fixed across levels. The block
# keeps its aspect ratio (nz = n_xy / 4). Basis sizes are fixed so that only h changes between levels.
X2_LEVELS = {
    "membrane": [lambda n=n: Membrane(n=n) for n in (15, 25, 40, 60)],
    "block": [lambda n=n: ElasticBlock(n_xy=n, nz=n // 4) for n in (8, 12, 16, 20)],
}
X2_SETTINGS = {
    "membrane": dict(n_train=100, n_test=100, r_primal=20, r_dual=24, dual_kind="nmf"),
    "block": dict(n_train=60, n_test=60, r_primal=15, r_dual=12, dual_kind="nmf"),
}

# Acceptance rule and fixed criteria (plan Sec. 6).
TAU = 0.05                       # relative certificate tolerance: accept if Delta <= TAU/(1+TAU) ||u^||_K
EFFECTIVITY_MAX_XK = 1.2         # X2: effectivity with X = K when Corollary E1(b) holds
S1_RESTRICTED_SHARPNESS = 1.5    # X4: S1-restricted within this factor of S1 ...
S1_RESTRICTED_SPEEDUP = 5.0      # ... at least this many times cheaper
FINITE_PRECISION_REL = 0.01      # X8: verified vs unverified certificate differ by < 1 %
GATE1_MIN_CERTIFIED_SHARE = 0.5  # Gate 1: speed claim needs certified share >= 50 % on the layered block

# Active-set prediction tolerances (plan Sec. 3, item 6), relative to the problem scales.
TAU_LAMBDA = 1e-8                # rule "multiplier": lam~_i > TAU_LAMBDA * ||lam~||_inf
TAU_GAP = 1e-8                   # rule "gap": (B u~ - c)_i <= TAU_GAP * char_gap
K_NEIGHBOURS = 3                 # rule "neighbours": union of training active sets of the k nearest parameters


def make_problem(name: str) -> ContactProblem:
    if name not in PROBLEMS:
        raise ValueError(f"unknown problem {name!r}; choose from {sorted(PROBLEMS)}")
    return PROBLEMS[name]()


def result_path(experiment: str, problem: str, seed: int) -> Path:
    return RESULTS / experiment / f"{problem}_seed{seed}.npz"
