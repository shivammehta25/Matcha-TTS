#!/usr/bin/env python
import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import torchaudio
from torch.hub import download_url_to_file
from tqdm import tqdm

from matcha.utils.data.utils import _extract_tar

URL = "https://www.openslr.org/resources/142/mcspeech.tar.gz"

INFO_PAGE = "https://www.openslr.org/142/"

LICENCE = "CC0 1.0"

CITATION = """
@masterthesis{mcspeech,
  title={Analiza porównawcza korpusów nagrań mowy dla celów syntezy mowy w języku polskim},
  author={Czyżnikiewicz, Mateusz},
  year={2022},
  month={December},
  school={Warsaw University of Technology},
  type={Master's thesis},
  doi={10.13140/RG.2.2.26293.24800},
  note={Available at \\url{http://dx.doi.org/10.13140/RG.2.2.26293.24800}},
}
"""


def decision():
    return random.random() < 0.98


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save-dir", type=str, default=None, help="Place to store the downloaded zip files")
    parser.add_argument(
        "-r",
        "--skip-resampling",
        action="store_true",
        default=False,
        help="Skip resampling the data (from 44.1 to 22.05)",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="data/mcspeech",
        help="Place to store the converted data, usually a subdirectory of data/",
    )

    return parser.parse_args()


def process_tsv(infile, outpath: Path):
    with (
        open(infile, encoding="utf-8") as inf,
        open(outpath / "train.tsv", "w", encoding="utf-8") as tf,
        open(outpath / "valid.tsv", "w", encoding="utf-8") as vf,
    ):
        for line in inf.readlines():
            line = line.strip()
            if line == "id\ttranscript":
                continue
            parts = line.split("\t")
            outfile = str(outpath / f"{parts[0]}.wav")
            if decision():
                tf.write(f"{outfile}|{parts[1]}\n")
            else:
                vf.write(f"{outfile}|{parts[1]}\n")


def process_files(tarfile, outpath, resample=True):
    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename in tqdm(_extract_tar(tarfile, tmpdirname)):
            if not filename.startswith(tmpdirname):
                filename = os.path.join(tmpdirname, filename)
            if filename.endswith(".tsv"):
                process_tsv(filename, outpath)
            else:
                filepart = filename.rsplit("/", maxsplit=1)[-1]
                outfile = str(outpath / filepart)
                arr, sr = torchaudio.load(filename)
                if resample:
                    arr = torchaudio.functional.resample(arr, orig_freq=sr, new_freq=22050)
                torchaudio.save(outfile, arr, 22050)


def main():
    args = get_args()

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir()

    if not args.output_dir:
        print("output directory not specified, exiting")
        sys.exit(1)

    outpath = Path(args.output_dir)
    if not outpath.is_dir():
        outpath.mkdir()

    resample = True
    if args.skip_resampling:
        resample = False

    if save_dir:
        tarname = URL.rsplit("/", maxsplit=1)[-1]
        tarfile = save_dir / tarname
        if not tarfile.exists():
            download_url_to_file(URL, str(tarfile), progress=True)
        process_files(str(tarfile), outpath, resample)
    else:
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=True) as zf:
            download_url_to_file(URL, zf.name, progress=True)
            process_files(zf.name, outpath, resample)


if __name__ == "__main__":
    main()
