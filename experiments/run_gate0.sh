#!/bin/sh
# Gate 0 runs (FINAL_RESEARCH_PLAN.md, Sec. 7): X1 on membrane / block / layered and X2 on membrane / block,
# seeds 0-2, active-set rules "multiplier" and "gap" (tau_gap = 1e-3). Logs go to results/logs/.
# Usage (from the project root):  sh experiments/run_gate0.sh
set -u
PY=.venv/Scripts/python.exe
SEEDS="0 1 2"
mkdir -p results/logs

run() {   # script, then its arguments
    script=$1; shift
    log="results/logs/$(basename "$script" .py)_$(echo "$*" | tr ' -' '_' | tr -s '_').log"
    $PY "$script" "$@" > "$log" 2>&1
    echo "exit=$? $script $*"
}

for seed in $SEEDS; do
    for problem in membrane block layered; do
        run experiments/x1_rate.py --problem $problem --seed $seed
        run experiments/x1_rate.py --problem $problem --seed $seed --rule gap --tau-gap 1e-3
    done
    for problem in membrane block; do
        run experiments/x2_effectivity.py --problem $problem --seed $seed --rule gap --tau-gap 1e-3
    done
done
python_summary="experiments/gate0_summary.py"
[ -f "$python_summary" ] && $PY "$python_summary" && echo "exit=$? gate0_summary"
