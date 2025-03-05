#!/bin/bash

### Evaluation ###

input_file= # Your test file
python_env= # Your Python Env
${python_env} -u utils/evaluate.py \
    --input-file ${input_file}
