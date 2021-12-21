We use a modified fork of [huggingface transformers](https://github.com/huggingface/transformers) for our experiments.

## Setup

```bash
$ git clone https://github.com/abhik1505040/crossum
$ cd crossum/seq2seq
$ conda create python==3.7.9 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch -p ./env
$ conda activate ./env # or source activate ./env (for older versions of anaconda)
$ bash setup.sh 
```

## Downloading data

This script downloads the metadata-stripped version of the dataset required for training.

```bash
$ bash download_data.sh
```

## Training

To see the list of all available options related to training, do `python pipeline.py -h`

### Running ablation experiments

* See available training settings: `bash trainer.sh -h`. You can try out different training hyperparameters by modifying the default values in this script.

Some sample commands for training on a 8 GPU node are given below. 
For multi-node usage with SLURM, refer to [job.sh]().

```bash
bash trainer.sh --ngpus 8 --training_type m2m # trains the many-to-many model
bash trainer.sh --ngpus 8 --training_type m2o --pivot_lang arabic # trains the many-to-one model using arabic as the target language
bash trainer.sh --ngpus 8 --training_type o2m --pivot_lang english # trains the one-to-many model using english as the source language
```

* Available pivot language names:
`oromo`, `french`, `amharic`, `arabic`, `azerbaijani`, `bengali`, `burmese`, `chinese_simplified`, `chinese_traditional`, `welsh`, `english`, `kirundi`, `gujarati`, `hausa`, `hindi`, `igbo`, `indonesian`, `japanese`, `korean`, `kyrgyz`, `marathi`, `spanish`, `scottish_gaelic`, `nepali`, `pashto`, `persian`, `pidgin`, `portuguese`, `punjabi`, `russian`, `serbian_cyrillic`, `serbian_latin`, `sinhala`, `somali`, `swahili`, `tamil`, `telugu`, `thai`, `tigrinya`, `turkish`, `ukrainian`, `urdu`, `uzbek`, `vietnamese`, `yoruba`

## Evaluation

* See available evaluation options: `python evaluator.py -h`. 
 
For example, to compute `ROUGE` and `LaSE` scores on all language pairs of the CrossSum test set using a trained cross-lingual model, run the following:

```bash
python evaluator.py \
    --dataset_dir <path/to/dataset/directory> \
    --output_dir <path/to/output/directory> \
    --evaluation_type xlingual \
    --data_type test \
    --xlingual_summarization_model_name_or_path <path/to/model/directory>
```

