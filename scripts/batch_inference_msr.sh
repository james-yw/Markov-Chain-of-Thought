#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${1:-0}



question_path= # Your question file
save_dir= # Your save dir
checkpoint_dir= # Your checkpoint dir

python_env= # Your Python Env
temperature= 0
seed= 1234

${python_env} -u mario/batch_react_inference_msr.py \
    --question_file ${question_path} \
    --save_dir ${save_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --temperature ${temperature} \
    --seed ${seed}
    