#!/bin/bash

ROOT_DATASET_DIR="dataset"
ROOT_MODEL_DIR="output"
RESULTS_DIR="evaluation_results"

for model_dir in $ROOT_MODEL_DIR/*/; do

    suffix=$(basename $model_dir)
    read training_type pivot_lang rest <<< $(IFS="_"; echo $suffix)

    if [[ "$training_type" = "m2o" ]]; then
        required_str="--required_tgt_lang ${pivot_lang}"
    elif [[ "$training_type" = "o2m" ]]; then
        required_str="--required_src_lang ${pivot_lang}"
    else
        required_str=" "
    fi

    for data_type in "val" "test"; do
        python evaluator.py \
            --dataset_dir "${ROOT_DATASET_DIR}" \
            --output_dir "${RESULTS_DIR}/${suffix}" \
            --evaluation_type xlingual \
            --data_type ${data_type} \
            --xlingual_summarization_model_name_or_path $model_dir \
            $required_str
    done
done