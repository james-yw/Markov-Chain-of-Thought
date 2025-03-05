#!/bin/bash
export CUDA_VISIBLE_DEVICES="${1:-0}"

python_env= #Your Python Env

${python_env} -u utils/construct_mcot_dataset.py