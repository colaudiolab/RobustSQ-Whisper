#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import logging
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description="format target speaker dataset.")
    parser.add_argument(
        "--in_dir", type=Path, required=True, help="Directory of the input dataset"
    )
    parser.add_argument(
        "--out_dir", type=Path, required=True, help="Directory of the output dataset"
    )

    return parser


def main():
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    # read wav.scp, text_spk1, text_spk2
    fin_wavscp = codecs.open(args.in_dir / "wav.scp", "r", "utf-8")
    fin_textspk1 = codecs.open(args.in_dir / "text_spk1", "r", "utf-8")
    fin_textspk2 = codecs.open(args.in_dir / "text_spk2", "r", "utf-8")

    # write wav.scp, text, utt2spk, spk2utt
    fout_wavscp = codecs.open(args.out_dir / "wav.scp", "w", "utf-8")
    fout_text = codecs.open(args.out_dir / "text", "w", "utf-8")
    fout_utt2spk = codecs.open(args.out_dir / "utt2spk", "w", "utf-8")
    fout_spk2utt = codecs.open(args.out_dir / "spk2utt", "w", "utf-8")

    for wavline, textline1, textline2 in zip(fin_wavscp, fin_textspk1, fin_textspk2):
        mixid, wavpath = wavline.strip().split(" ")
        text1 = textline1.strip().split(" ", 1)[1]
        text2 = textline2.strip().split(" ", 1)[1]

        # �?mixid 中提取真实的 speaker ID
        # 例如: 103-1240-0003_1235-135887-0017 -> spk1: 103-1240-0003, spk2: 1235-135887
        mixid_parts = mixid.split("_")
        spk1_id = mixid_parts[0]  # 第一个说话人ID
        spk2_id = mixid_parts[1]  # 第二个说话人ID

        # 修改utterance ID格式，让speaker ID作为前缀以满足排序要�?
        utt1_id = f"{spk1_id}_{mixid}_spk1"
        utt2_id = f"{spk2_id}_{mixid}_spk2"

        # 修正路径：将错误的绝对路径替换为正确的相对路�?
        corrected_wavpath = wavpath.replace("/path/to/your/LibriMix_metadata/", "data/")
        
        # write wav.scp with new utterance IDs and corrected paths
        fout_wavscp.write(f"{utt1_id} {corrected_wavpath}\n")
        fout_wavscp.write(f"{utt2_id} {corrected_wavpath}\n")

        # write text with new utterance IDs
        fout_text.write(f"{utt1_id} {text1}\n")
        fout_text.write(f"{utt2_id} {text2}\n")

        # write utt2spk with proper speaker IDs
        fout_utt2spk.write(f"{utt1_id} {spk1_id}\n")
        fout_utt2spk.write(f"{utt2_id} {spk2_id}\n")

        # write spk2utt will be generated later using utils/utt2spk_to_spk2utt.pl

    fout_wavscp.close()
    fout_text.close()
    fout_utt2spk.close()
    fout_spk2utt.close()

    fin_wavscp.close()
    fin_textspk1.close()
    fin_textspk2.close()

    # Generate spk2utt from utt2spk using standard method
    import subprocess
    import sys
    
    try:
        # Use ESPnet's utt2spk_to_spk2utt.pl to generate spk2utt
        result = subprocess.run([
            "utils/utt2spk_to_spk2utt.pl", 
            str(args.out_dir / "utt2spk")
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            with open(args.out_dir / "spk2utt", "w") as f:
                f.write(result.stdout)
            logging.info(f"Generated spk2utt for {args.out_dir}")
        else:
            logging.warning(f"Failed to generate spk2utt: {result.stderr}")
            # Fallback: generate spk2utt manually
            generate_spk2utt_manually(args.out_dir)
            
    except Exception as e:
        logging.warning(f"Error running utt2spk_to_spk2utt.pl: {e}")
        # Fallback: generate spk2utt manually
        generate_spk2utt_manually(args.out_dir)


def generate_spk2utt_manually(out_dir):
    """Generate spk2utt manually from utt2spk"""
    spk2utt = {}
    with open(out_dir / "utt2spk", "r") as f:
        for line in f:
            utt, spk = line.strip().split()
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt)
    
    with open(out_dir / "spk2utt", "w") as f:
        for spk in sorted(spk2utt.keys()):
            utts = " ".join(sorted(spk2utt[spk]))
            f.write(f"{spk} {utts}\n")


if __name__ == "__main__":
    main()
