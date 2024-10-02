#!/usr/bin/env python
import argparse
import random
import sys
import tempfile
from pathlib import Path

from torch.hub import download_url_to_file

from matcha.utils.data.utils import _extract_tar

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

INFO_PAGE = "https://keithito.com/LJ-Speech-Dataset/"

LICENCE = "Public domain (LibriVox copyright disclaimer)"

CITATION = """
@misc{ljspeech17,
  author       = {Keith Ito and Linda Johnson},
  title        = {The LJ Speech Dataset},
  howpublished = {\\url{https://keithito.com/LJ-Speech-Dataset/}},
  year         = 2017
}
"""


def decision():
    return random.random() < 0.98


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save-dir", type=str, default=None, help="Place to store the downloaded zip files")
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="data/LJSpeech-1.1",
        help="Place to store the converted data, usually a subdirectory of data/",
    )

    return parser.parse_args()


def main():
    args = get_args()

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir()

    outpath = Path(args.output_dir)
    if not outpath.is_dir():
        outpath.mkdir()

    if save_dir:
        tarname = URL.rsplit("/", maxsplit=1)[-1]
        tarfile = str(save_dir / tarname)
        _extract_tar(tarfile, outpath)
    else:
        with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=True) as zf:
            download_url_to_file(URL, zf.name, progress=True)
            _extract_tar(zf.name, outpath)


if __name__ == "__main__":
    main()
