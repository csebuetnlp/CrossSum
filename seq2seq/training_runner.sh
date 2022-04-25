#!/bin/bash

bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang english
bash trainer.sh --ngpus 8 --training_type o2m --pivot_lang english
bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang hindi
bash trainer.sh --ngpus 8 --training_type o2m --pivot_lang hindi
bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang russian
bash trainer.sh --ngpus 8 --training_type o2m --pivot_lang russian
bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang arabic
bash trainer.sh --ngpus 8 --training_type o2m --pivot_lang arabic
