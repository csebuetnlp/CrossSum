#!/bin/bash

python evaluator.py \
    --dataset_dir dataset/ \
    --output_dir evaluation_results/baseline \
    --evaluation_type baseline \
    --data_type test \
    --required_src_lang russian 
    # --xlingual_summarization_model_name_or_path /home/rifat/Documents/crossum/seq2seq/output/o2m_hindi_with_native/checkpoint-25000
