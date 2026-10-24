#!/bin/bash
# Usage: bash scripts/build_swisscrawl_testset.sh [--resume]
set -euo pipefail

python src/score_candidates.py \
    --path_candidates data/swisscrawl_filtered.csv \
    --path_scored data/swisscrawl_scored_gahd.csv \
    --batch_size 32 \
    "$@"

python src/sample_master.py \
    --path_scored data/swisscrawl_scored_gahd.csv \
    --path_master data/master_sample_stratified.csv \
    --path_sampling_report data/stratified_sampling_report.json \
    --seed 42 \
    "$@"

python src/translate_master.py \
    --path_master data/master_sample_stratified.csv \
    --path_translation_checkpoint data/checkpoints/haiku_batch_translation.json \
    --num_api_attempts 5 \
    "$@"

python src/build_annotation_files.py \
    --path_master data/master_sample_stratified.csv \
    --path_annotation_dir data \
    --seed 42 \
    "$@"

python src/verify_testset.py \
    --path_master data/master_sample_stratified.csv \
    --path_annotation_dir data \
    --path_verification_report data/stratified_verification_report.json \
    "$@"
