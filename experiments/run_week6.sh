#!/bin/sh
# Week 6: X3 (all seeds) -> rule selection (x3_select.py, fixed rule) -> X4 (all seeds) with the selected tau_g.
# Usage (from the project root):  sh experiments/run_week6.sh
set -u
PY=.venv/Scripts/python.exe
PROBLEMS="membrane block layered membrane_shift_mild"
mkdir -p results/logs
for seed in 0 1 2; do
    for p in $PROBLEMS; do
        [ -f "results/x3/${p}_seed${seed}.npz" ] && continue
        $PY experiments/x3_active_set.py --problem $p --seed $seed > results/logs/x3_${p}_seed${seed}.log 2>&1
        echo "exit=$? x3 $p seed $seed"
    done
done
$PY experiments/x3_select.py > results/logs/x3_select.log 2>&1
echo "exit=$? x3_select"
TAU=$($PY -c "import json; s=json.load(open('results/x3/selection.json'))['selected']; print(s.split('_')[1] if s.startswith('gap_') else '')")
if [ -z "$TAU" ]; then echo "selected rule is not a gap rule; X4 not run"; exit 1; fi
echo "selected tau_gap = $TAU"
for seed in 0 1 2; do
    for p in $PROBLEMS; do
        $PY experiments/x4_multipliers.py --problem $p --seed $seed --tau-gap $TAU > results/logs/x4_${p}_seed${seed}.log 2>&1
        echo "exit=$? x4 $p seed $seed"
    done
done
