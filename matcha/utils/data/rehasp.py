#!/usr/bin/env python
import argparse
import os
import random
import tempfile
from pathlib import Path

import torchaudio
from torch.hub import download_url_to_file
from tqdm import tqdm

from matcha.utils.data.utils import _extract_zip

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/561/rehasp_0.5.zip?sequence=4&isAllowed=y"

INFO_PAGE = "https://datashare.ed.ac.uk/handle/10283/561"

LICENCE = "Creative Commons Attribution 4.0 International"

CITATION = """
@inproceedings{henter14_interspeech,
  title     = {Measuring the perceptual effects of modelling assumptions in speech synthesis using stimuli constructed from repeated natural speech},
  author    = {Gustav Eje Henter and Thomas Merritt and Matt Shannon and Catherine Mayo and Simon King},
  year      = {2014},
  booktitle = {Interspeech 2014},
  pages     = {1504--1508},
  doi       = {10.21437/Interspeech.2014-361},
  issn      = {2958-1796},
}"""


def decision():
    return random.random() < 0.90


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--save-dir", type=str, default=None, help="Place to store the downloaded zip file")
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="data",
        help="Place to store the converted data (subdirectory rehasp_0.5 will be created)",
    )

    return parser.parse_args()


def process_data(zipfile, rehasppath: Path, resample=True):
    def slurp(filename):
        with open(filename) as inf:
            return inf.read().strip()

    prompts = {}
    wavfiles = []

    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename in tqdm(_extract_zip(zipfile, tmpdirname)):
            stem = filename.rsplit("/", maxsplit=1)[1].rsplit(".", maxsplit=1)[0]
            if not filename.startswith(tmpdirname):
                filename = os.path.join(tmpdirname, filename)
            if "96k/lucy" in filename and filename.endswith(".wav"):
                outfile = rehasppath / f"{stem}.wav"
                arr, sr = torchaudio.load(filename)
                if resample:
                    arr = torchaudio.functional.resample(arr, orig_freq=sr, new_freq=22050)
                torchaudio.save(outfile, arr, 22050)
                wavfiles.append(str(outfile))
            elif "prompts" in filename and filename.endswith(".txt"):
                text = slurp(filename)
                prompts[stem] = text

    with (
        open(rehasppath / "train.txt", "w", encoding="utf-8") as tf,
        open(rehasppath / "val.txt", "w", encoding="utf-8") as vf,
    ):
        for wavfile in wavfiles:
            wavstem = wavfile.rsplit("/", maxsplit=1)[1].rsplit(".", maxsplit=1)[0]
            parts = wavstem.split("_")
            text = prompts[parts[3]]

            if decision():
                tf.write(f"{wavfile}|{text}\n")
            else:
                vf.write(f"{wavfile}|{text}\n")


def main():
    args = get_args()

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir()

    dirname = "rehasp_0.5"
    outbasepath = Path(args.output_dir)
    if not outbasepath.is_dir():
        outbasepath.mkdir()
    outpath = outbasepath / dirname
    if not outpath.is_dir():
        outpath.mkdir()

    if save_dir:
        zipname = URL.rsplit("/", maxsplit=1)[-1].split("?", maxsplit=1)[0]
        zipfile = save_dir / zipname
        if not zipfile.exists():
            download_url_to_file(URL, str(zipfile), progress=True)
        process_data(zipfile, outpath)
    else:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as zf:
            download_url_to_file(URL, zf.name, progress=True)
            process_data(zf.name, outpath)


if __name__ == "__main__":
    main()
