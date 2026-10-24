#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/gahd_gbert_raw_run42.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/gahd_gbert_raw_run43.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/gahd_gbert_raw_run44.json
