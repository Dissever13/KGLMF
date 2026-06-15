#!/bin/bash
# Batch run all KG-LMF experiments
# Usage: bash experiments/scripts/run_all_experiments.sh

PYTHON=/d/anaconda3/envs/kglmf/python
SCRIPT=experiments/scripts/run_full_experiment.py
OUTDIR=results/
N=81

cd D:/KGLM/KGLMF

echo "=== Starting full experiment batch ==="
echo "Datasets: triviaqa, strategyqa, popqa, kodcode, bigcodebench, pddl, hotpotqa, coqa"
echo "N=$N per dataset per method"
echo ""

for DS in popqa kodcode bigcodebench pddl hotpotqa coqa; do
    echo ""
    echo "===== $DS ====="
    $PYTHON $SCRIPT --method both --n_samples $N --dataset $DS --output_dir $OUTDIR
    echo "===== $DS DONE ====="
done

echo ""
echo "=== All experiments complete ==="
