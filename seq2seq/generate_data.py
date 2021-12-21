import os
import glob
import shutil
import argparse
from tqdm import tqdm

def get_lc(path):
    i = 0
    with open(path) as f:
        for i, _ in enumerate(f, 1):
            pass

    return i

def run(args):
    data_type = "train"
    type2prefix = {
        "m2m": "*",
        "m2o": f"*-{args.pivot_lang}",
        "o2m": f"{args.pivot_lang}-*"
    }
    iterator = glob.glob(
        os.path.join(
            args.dataset_dir,
            type2prefix[args.training_type] + "_" + data_type + "*.target"
        )
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for f in tqdm(iterator, desc="Moving data files for training"):
        basename = os.path.basename(f)
        src_lang, tgt_lang = basename.rsplit("_", 1)[0].split("-")

        if src_lang == tgt_lang and args.exclude_native == "yes":
            continue

        if get_lc(f) >= args.min_example_count:
            shutil.copy(
                os.path.join(
                    args.dataset_dir,
                    src_lang + "-" + tgt_lang + "_" + data_type + ".source"
                ),
                os.path.join(
                    args.output_dir,
                    src_lang + "-" + tgt_lang + "_" + data_type + ".source"
                )
            )

            shutil.copy(
                os.path.join(
                    args.dataset_dir,
                    src_lang + "-" + tgt_lang + "_" + data_type + ".target"
                ),
                os.path.join(
                    args.output_dir,
                    src_lang + "-" + tgt_lang + "_" + data_type + ".target"
                )
            )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        metavar='PATH',
        help="Input directory")

    parser.add_argument(
        '--output_dir',
        required=True,
        metavar='PATH',
        help="Output directory")

    parser.add_argument(
        '--pivot_lang', type=str,
        required=True,
        help="Pivot language")

    parser.add_argument(
        '--training_type', type=str, 
        choices=["m2m", "m2o", "o2m"],
        required=True, 
        help='Training type (many-to-many/many-to-one/one-to-many)'
    )
    parser.add_argument('--exclude_native', type=str,
        default=False, 
        help='Exclude the native-to-native filepairs during training'
    )
    parser.add_argument('--min_example_count', type=int,
        default=32, 
        help='Minimum example count for a training pair to be included in training'
    )

    args = parser.parse_args()
    run(args)