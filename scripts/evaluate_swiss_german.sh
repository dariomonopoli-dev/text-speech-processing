#!/bin/bash
set -e

SEEDS="42 43 44"

evaluate_approach() {
    experiment=$1
    input=$2
    for seed in $SEEDS; do
        path_run=runs/$experiment/run$seed
        if [ ! -f "$path_run/predictions_swiss_german_$input.csv" ]; then
            echo "Skip $experiment ($input): no predictions in $path_run"
            return 0
        fi
        python3 src/evaluate_predictions.py \
            --path_predictions "$path_run/predictions_swiss_german_$input.csv" \
            --path_metrics "$path_run/swiss_german_${input}_metrics.json" \
            --overwrite
    done
    python3 src/parse_results_into_csv.py \
        --path_runs runs/$experiment \
        --metrics_pattern "swiss_german_${input}_metrics.json" \
        --path_output_dir runs/$experiment/swiss_german_$input \
        --predictions_name "predictions_swiss_german_$input.csv" \
        --bootstrap
}

python3 src/compute_annotation_agreement.py
evaluate_approach gahd_gbert_raw text
evaluate_approach gahd_gbert_raw translation_haiku
evaluate_approach gahd_gbert_raw translation_apertus8b
evaluate_approach gahd_swissbert_raw text
