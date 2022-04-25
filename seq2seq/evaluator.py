import os
import gc
import glob
import shutil
import argparse
import logging
import json
from tqdm import tqdm
import numpy as np
from generate_data import get_lc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from LaSE import LaSEScorer 
from LaSE.utils import LANG2ISO
from utils import calculate_rouge

logging.basicConfig(level=logging.INFO)


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        metavar='PATH',
        help="Input directory"
    )

    parser.add_argument(
        '--output_dir',
        required=True,
        metavar='PATH',
        help="Output directory"
    )

    parser.add_argument(
        '--evaluation_type', type=str, 
        choices=["baseline", "xlingual"],
        required=True, 
        help="""Evaluation type 
                (baseline i.e. summarization + translation /
                cross-lingual summarization)"""
    )

    parser.add_argument(
        '--data_type', type=str, 
        choices=["val", "test"],
        required=True, 
        help="""Evaluation data type (validation / test)"""
    )

    parser.add_argument(
        '--xlingual_summarization_model_name_or_path',
        metavar='PATH',
        help="""HF name or path to cross-lingual summarization model.
                Applicable when evaluation_type == xlingual"""
    )

    parser.add_argument(
        '--multilingual_summarization_model_name_or_path',
        metavar='PATH',
        default="csebuetnlp/mT5_multilingual_XLSum",
        help="""HF name or path to multi-lingual summarization model.
                Applicable when evaluation_type == baseline"""
    )

    parser.add_argument(
        '--multilingual_translation_model_name_or_path',
        metavar='PATH',
        default="facebook/m2m100_418M",
        help="""HF name or path to multi-lingual translation model.
                Applicable when evaluation_type == baseline"""
    )

    parser.add_argument('--max_source_length', type=int,
        default=512, 
        help='Maximum source length'
    )

    parser.add_argument('--max_target_length', type=int,
        default=84, 
        help='Maximum target length'
    )

    parser.add_argument('--batch_size', type=int,
        default=16, 
        help='Evaluation batch size'
    )

    parser.add_argument('--beam_size', type=int,
        default=5, 
        help='Evaluation beam size'
    )

    parser.add_argument('--no_repeat_ngram_size', type=int,
        default=2, 
        help='Evaluation no repeat ngram size'
    )

    parser.add_argument('--length_penalty', type=float,
        default=0.6, 
        help='Evaluation length penalty'
    )

    parser.add_argument('--required_src_lang', type=str,
        default=None, 
        help='''Only evaluate pairs having this language as the src.
                If not any of required_src_lang, required_tgt_lang and required_pairs
                are provided, runs evaluation on all found pairs.'''
    )

    parser.add_argument('--required_tgt_lang', type=str,
        default=None, 
        help='''Only evaluate pairs having this language as the tgt.
                If not any of required_src_lang, required_tgt_lang and required_pairs
                are provided, runs evaluation on all found pairs.'''
    )

    parser.add_argument('--required_pairs', type=str, nargs="*",
        default=[], 
        help='''Only evaluate these language pairs. Pair names have to be hyphenated (e.g. `bengali-english`).
                If not any of required_src_lang, required_tgt_lang and required_pairs
                are provided, runs evaluation on all found pairs.'''
    )
    
    parser.add_argument('--device', type=str,
        default="cuda", 
        help='''Evaluation device'''
    )

    return parser

def read_json(path):
    with open(path) as f:
        return json.load(f)

def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def get_batches(data_iterator, batch_size=8):
    for i in range(0, len(data_iterator), batch_size):
        yield data_iterator[i: i + batch_size]

def read_lines(input_path):
    with open(input_path) as f:
        return [l.strip() for l in f.readlines()]

_LOADED_MODELS = {}
_LASE_SCORER = LaSEScorer()

def load_model(model_name_or_path, model_type, device):
    global _LOADED_MODELS
    if model_type not in _LOADED_MODELS:
        _LOADED_MODELS[model_type] = {
            "tokenizer": AutoTokenizer.from_pretrained(model_name_or_path),
            "model": AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
        }

def summarize_and_translate(
    input_path,
    summarization_path,
    translation_path,
    src_lang,
    tgt_lang,
    args
):

    global _LOADED_MODELS
    
    if os.path.isfile(summarization_path) and os.path.isfile(translation_path):
        return

    # run summarization
    input_lines = read_lines(input_path)

    summarized_lines = []
    with open(summarization_path, 'w') as outf:
        for batch in get_batches(input_lines, batch_size=args.batch_size):
            encoded_tokens = _LOADED_MODELS["summarization"]["tokenizer"](
                batch, 
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=args.max_source_length
            ).to(args.device)
            
            generated_tokens = _LOADED_MODELS["summarization"]["model"].generate(
                **encoded_tokens,
                max_length=args.max_target_length,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                num_beams=args.beam_size,
                length_penalty=args.length_penalty
            )

            output_lines = _LOADED_MODELS["summarization"]["tokenizer"].batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

            summarized_lines += output_lines

            for o in output_lines:
                print(o.strip(), file=outf)
    
    # run translation
    _LOADED_MODELS["translation"]["tokenizer"].src_lang = src_lang
    
    with open(translation_path, 'w') as outf:
        for batch in get_batches(summarized_lines, batch_size=args.batch_size):
            encoded_tokens = _LOADED_MODELS["translation"]["tokenizer"](
                batch, 
                return_tensors="pt",
                padding="longest"
            ).to(args.device)
            
            generated_tokens = _LOADED_MODELS["translation"]["model"].generate(
                **encoded_tokens,
                forced_bos_token_id=_LOADED_MODELS["translation"]["tokenizer"].get_lang_id(tgt_lang)
            )
            output_lines = _LOADED_MODELS["translation"]["tokenizer"].batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

            for o in output_lines:
                print(o.strip(), file=outf)

def summarize_xlingual(
    input_dir,
    output_dir,
    tgt_lang,
    args
):
    if os.path.isfile(os.path.join(output_dir, f"{args.data_type}_generations.txt")):
        return

    script_path =  os.path.abspath("pipeline.py")
    script_args = [
        f"--model_name_or_path {args.xlingual_summarization_model_name_or_path}",
        f"--data_dir {input_dir}",
        f"--output_dir {output_dir}",
        f"--per_device_eval_batch_size {args.batch_size}",
        f"--max_source_length {args.max_source_length}",
        f"--{args.data_type}_max_target_length {args.max_target_length}",
        f"--length_penalty {args.length_penalty}",
        f"--no_repeat_ngram_size {args.no_repeat_ngram_size}",
        f"--eval_beams {args.beam_size}",
        f"--tgt_lang {tgt_lang}",
        f"--rouge_lang {tgt_lang}",
        "--overwrite_output_dir",
        "--predict_with_generate",
        "--do_predict",
        "--use_langid",
        "--seed 1234"        
    ]

    cmd = "python " + script_path + " " + " ".join(script_args)
    os.system(cmd)


def calculate_lase(
    pred_lns,
    tgt_lns,
    tgt_lang
):
    global _LASE_SCORER
    scores = [_LASE_SCORER.score(ref.strip(), pred.strip(), target_lang=tgt_lang) 
                    for ref, pred in zip(tgt_lns, pred_lns)]

    return {
        "LaSE": round(np.mean(scores) * 100, 4)
    }


def run(args):
    root_output_dir = os.path.join(args.output_dir, args.data_type, "outputs")
    root_log_dir = os.path.join(args.output_dir, args.data_type, "logs")

    os.makedirs(root_output_dir, exist_ok=True)
    os.makedirs(root_log_dir, exist_ok=True)

    if args.evaluation_type == "baseline":
        load_model(
            args.multilingual_summarization_model_name_or_path,
            "summarization",
            args.device
        )
        load_model(
            args.multilingual_translation_model_name_or_path,
            "translation",
            args.device
        )
    
    source_suffix = f"_{args.data_type}.source"
    target_suffix = f"_{args.data_type}.target"

    required_files = glob.glob(os.path.join(args.dataset_dir, "*" + target_suffix))
    required_pairs = [os.path.basename(k).rsplit(target_suffix, 1)[0] for k in required_files]
    
    if args.required_pairs:
        required_pairs = [k for k in args.required_pairs
                            if os.path.isfile(os.path.join(args.dataset_dir, k + target_suffix))]

    if args.required_src_lang:
        required_pairs = [k for k in required_pairs
                            if k.split("-")[0] == args.required_src_lang]

    if args.required_tgt_lang:
        required_pairs = [k for k in required_pairs
                            if k.split("-")[1] == args.required_tgt_lang]

    required_pairs = sorted(required_pairs)
    
    for pair in tqdm(required_pairs, desc="Running evaluation"):
        src_lang, tgt_lang = pair.split("-")
        scores = {}
        log_path = os.path.join(root_log_dir, pair + ".log")

        if os.path.isfile(log_path):
            continue

        def evaluate(lase_key):
            dir_prefix = pair + "-" + ("crossum" if lase_key == "LaSE_in_lang" else "xlsum")
            dir_name = os.path.join(root_output_dir, dir_prefix)
            os.makedirs(dir_name, exist_ok=True)
            
            prefix = pair if lase_key == "LaSE_in_lang" else f"{src_lang}-{src_lang}"
            root_source_path = os.path.join(args.dataset_dir, prefix + source_suffix)
            pipeline_source_path = os.path.join(dir_name, source_suffix[1:])
            root_target_path = os.path.join(args.dataset_dir, prefix + target_suffix)
            pipeline_target_path = os.path.join(dir_name, target_suffix[1:])

            if (
                    not os.path.isfile(root_source_path) or
                    not os.path.isfile(root_target_path) or
                    get_lc(root_target_path) == 0
            ):
                return
            
            shutil.copy(
                root_source_path,
                pipeline_source_path
            )

            shutil.copy(
                root_target_path,
                pipeline_target_path
            )

            # specially handly validation files
            # since output file is generated for 
            # test files only
            if args.data_type == "val":
                shutil.copy(
                    pipeline_source_path,
                    os.path.join(dir_name, "test.source")
                )
                shutil.copy(
                    pipeline_source_path,
                    os.path.join(dir_name, "test.target")
                )

            if args.evaluation_type == "xlingual":
                summarize_xlingual(dir_name, dir_name, tgt_lang, args)

                if args.data_type == "val":
                    shutil.move(
                        os.path.join(dir_name, f"test_generations.txt"),
                        os.path.join(dir_name, f"val_generations.txt")
                    )

                    os.remove(os.path.join(dir_name, "test.source"))
                    os.remove(os.path.join(dir_name, "test.target"))

                pred_lines = read_lines(
                    os.path.join(dir_name, f"{args.data_type}_generations.txt")
                )
                ref_lines = read_lines(pipeline_target_path)



            elif args.evaluation_type == "baseline":
                src_iso, tgt_iso = LANG2ISO.get(src_lang, None), LANG2ISO.get(tgt_lang, None)
                if (
                        not src_iso or 
                        not tgt_iso or 
                        src_iso not in _LOADED_MODELS["translation"]["tokenizer"].lang_code_to_token or
                        tgt_iso not in _LOADED_MODELS["translation"]["tokenizer"].lang_code_to_token
                ):
                    return

                summarized_path = pipeline_source_path + ".summarized"
                translated_path = summarized_path + ".translated"

                summarize_and_translate(
                    pipeline_source_path,
                    summarized_path,
                    translated_path,
                    src_iso,
                    tgt_iso,
                    args
                )

                pred_lines = read_lines(translated_path)
                ref_lines = read_lines(pipeline_target_path)

            if lase_key == "LaSE_in_lang":
                scores.update(
                    calculate_rouge(pred_lines, ref_lines, rouge_lang=tgt_lang)
                )

            lase_scores = calculate_lase(pred_lines, ref_lines, tgt_lang)
            scores[lase_key] = lase_scores["LaSE"]

            
        # first do crossum evaluation (in lang LaSE)
        evaluate("LaSE_in_lang")

        if src_lang != tgt_lang:
            # now do xlsum evaluation (out lang LaSE)
            evaluate("LaSE_out_lang")
            
        # write combined results
        write_json(
            scores, 
            log_path
        )

        gc.collect()

    # aggregate results
    combined_results_path = os.path.join(args.output_dir, args.data_type, "combined_results.log")
    logging.info("Writing the combined results to " + combined_results_path)

    with open(combined_results_path, 'w') as outf:
        iterator = glob.glob(
            os.path.join(root_log_dir, "*.log")
        )
        
        keys = ["Language pair", "rouge1", "rouge2", "rougeL", "LaSE_in_lang", "LaSE_out_lang"]
        row_format = "{}\t" * (len(keys) - 1) + "{}"
        
        header = row_format.format(*keys)
        print(header, file=outf)
        
        for log_path in iterator:
            data = read_json(log_path)
            lang_pair = os.path.basename(log_path).rsplit(".log", 1)[0]
            row = [lang_pair] + [data.get(k, "") for k in keys[1:]]
            print(row_format.format(*row), file=outf)





if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run(args)
