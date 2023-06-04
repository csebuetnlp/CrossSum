#!/bin/bash

CHECKPOINT_DIR="finetuned_models"

bash trainer.sh --ngpus 8 --training_type m2m --minibatching fixed_tgt --per_lang_batch_size 32 --copy_last_checkpoint $CHECKPOINT_DIR
bash trainer.sh --ngpus 8 --training_type m2m --minibatching fixed_src --per_lang_batch_size 32 --copy_last_checkpoint $CHECKPOINT_DIR
bash trainer.sh --ngpus 8 --training_type m2m --per_lang_batch_size 32 --copy_last_checkpoint $CHECKPOINT_DIR
bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang english --copy_last_checkpoint $CHECKPOINT_DIR
bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang hindi --copy_last_checkpoint $CHECKPOINT_DIR
bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang russian --copy_last_checkpoint $CHECKPOINT_DIR
bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang arabic --copy_last_checkpoint $CHECKPOINT_DIR
bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang chinese_simplified --copy_last_checkpoint $CHECKPOINT_DIR

tar -cjvf "$CHECKPOINT_DIR.tar.bz2" $CHECKPOINT_DIR
