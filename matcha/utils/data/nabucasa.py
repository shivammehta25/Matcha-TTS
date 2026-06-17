import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import torchaudio
from torch.hub import download_url_to_file

from matcha.utils.data.utils import _extract_zip

INFO = "https://github.com/NabuCasa/voice-datasets"

LICENCE = "CC0 (public domain)"

URLS = {
    "bg_BG": {"dimitar": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/bg_BG-dimitar.zip"},
    "de_DE": {"kerstin": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/de_DE-kerstin.zip"},
    "en_US": {
        "joe": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/en_US-joe.zip",
        "kathleen": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/en_US-kathleen.zip",
    },
    "es_ES": {
        "dave": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/es_ES-dave.zip",
    },
    "hu_HU": {
        "anna": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/hu_HU-anna.zip",
        "berta": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/hu_HU-berta.zip",
        "imre": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/hu_HU-imre.zip",
    },
    "it_IT": {"paola": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/it_IT-paola.zip"},
    "nl_BE": {
        "flemishguy": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/nl_BE-flemishguy.zip",
        "nathalie": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/nl_BE-nathalie.zip",
    },
    "pl_PL": {
        "darkman": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/pl_PL-darkman.zip",
        "gosia": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/pl_PL-gosia.zip",
    },
    "pt_BR": {"faber": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/pt_BR-faber.zip"},
    "pt_PT": {"tugão": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/pt_PT-tugao.zip"},
    "ro_RO": {"mihai": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/ro_RO-mihai.zip"},
    "ru_RU": {
        "denis": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/ru_RU-denis.zip",
        "dmitri": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/ru_RU-dmitri.zip",
    },
    "sk_SK": {"lili": "https://github.com/NabuCasa/voice-datasets/releases/download/v1.0.0/sk_SK-lili.zip"},
}


def decision():
    return random.random() < 0.98


def get_languages():
    langs = {}
    for k in URLS:
        langcode = k.split("_")[0]
        if not langcode in langs:
            langs[langcode] = set()
        langs[langcode].add(k)
    return {k: list(v) for k, v in langs.items()}


def get_per_voice():
    output = {}
    for lang in URLS:  # pylint: disable=consider-using-dict-items
        for voice in URLS[lang]:
            output[voice] = URLS[lang][voice]
    # only one non-ASCII name
    output["tugao"] = output["tugão"]
    return output


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l", "--language", type=str, help="The name of the language to prepare (prepares all available voices)"
    )
    parser.add_argument("-v", "--voice", type=str, help="The name of the voice to prepare (single voice)")
    parser.add_argument(
        "-V", "--list-voices", type=str, default=None, help="List available voices for specified language"
    )
    parser.add_argument(
        "-L", "--list-languages", action="store_true", default=False, required=False, help="List available languages"
    )
    parser.add_argument("-s", "--save-dir", type=str, default=None, help="Place to store the downloaded zip files")
    parser.add_argument("-o", "--output-dir", type=str, default="data", help="Place to store the converted data")

    return parser.parse_args()


def print_languages(languages, concise=True):
    print("The following languages are available:")
    print(", ".join(languages.keys()))
    for lang in languages:
        if len(languages[lang]) != 1:
            print(f"{lang} has the sublanguages {', '.join(languages[lang])}")
        elif not concise:
            print(f"{lang} has the sublanguage {languages[lang][0]}")


def _get_voice_names(language, languages):
    if "_" in language:
        if language not in URLS:
            return []
        return URLS[language]
    else:
        if language not in languages:
            return []
        voices = []
        for sublang in languages[language]:
            voices += URLS[sublang].keys()
        return voices


def convert_zip_contents(filename, outpath, resample=True):
    with (
        tempfile.TemporaryDirectory() as tmpdirname,
        open(outpath / "train.txt", "w", encoding="utf-8") as tf,
        open(outpath / "valid.txt", "w", encoding="utf-8") as vf,
    ):
        for file in _extract_zip(filename, tmpdirname):
            if not file.startswith(tmpdirname):
                file = os.path.join(tmpdirname, file)
            filepart = file.rsplit("/", maxsplit=1)[-1]
            if file.endswith(".webm"):
                outfile = str(outpath / filepart).replace(".webm", ".wav")
                arr, sr = torchaudio.load(file)
                out_sr = sr
                if resample:
                    arr = torchaudio.functional.resample(arr, orig_freq=sr, new_freq=22050)
                    out_sr = 22050
                torchaudio.save(outfile, arr, out_sr)

            elif file.endswith(".flac"):
                outfile = str(outpath / filepart).replace(".flac", ".wav")
                arr, sr = torchaudio.load(file)
                out_sr = sr
                if resample:
                    arr = torchaudio.functional.resample(arr, orig_freq=sr, new_freq=22050)
                    out_sr = 22050
                torchaudio.save(outfile, arr, out_sr)

            elif file.endswith(".txt"):
                outfile = str(outpath / filepart).replace(".txt", ".wav")
                with open(file, encoding="utf-8") as txtf:
                    text = txtf.read().strip()
                if decision():
                    tf.write(f"{outfile}|{text}\n")
                else:
                    vf.write(f"{outfile}|{text}\n")

            else:
                continue


def merge_voices(voices, datapath, outpath):
    with (
        open(outpath / "train.txt", "w", encoding="utf-8") as tf,
        open(outpath / "valid.txt", "w", encoding="utf-8") as vf,
    ):
        for i, v in enumerate(voices, 1):
            voicepath = datapath / f"nabucasa_{v}"
            with (
                open(voicepath / "train.txt", encoding="utf-8") as t_inf,
                open(voicepath / "valid.txt", encoding="utf-8") as v_inf,
            ):
                for line in t_inf.readlines():
                    parts = line.strip().split("|")
                    tf.write(f"{parts[0]}|{i}|{parts[1]}\n")
                for line in v_inf.readlines():
                    parts = line.strip().split("|")
                    vf.write(f"{parts[0]}|{i}|{parts[1]}\n")


def main():
    args = get_args()

    languages = get_languages()
    if args.list_languages:
        print_languages(languages, True)
        sys.exit(1)

    if args.list_voices:
        voices = _get_voice_names(args.list_voices, languages)
        if voices == []:
            print("Language", args.list_voices, "not available")
            print_languages(languages, True)
            sys.exit(1)
        print(", ".join(voices))

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir()

    def process_single(voice, outdir, all_voices, save_dir=None, resample=True):
        url = all_voices[voice]
        zipfilename = url.split("/")[-1]
        voicedir = Path(outdir) / f"nabucasa_{voice}"
        if not voicedir.is_dir():
            voicedir.mkdir()
        if save_dir:
            zfile = str(save_dir / zipfilename)
            if not (save_dir / zipfilename).exists():
                download_url_to_file(all_voices[voice], zfile, progress=True)
            convert_zip_contents(zfile, voicedir, resample)
        else:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as zf:
                download_url_to_file(all_voices[voice], zf.name, progress=True)
                convert_zip_contents(zf.name, voicedir, resample)

    all_voices = get_per_voice()
    if args.voice:
        if args.voice not in all_voices:
            print("Voice", args.voice, "not available")
            sys.exit(1)
        process_single(args.voice, args.output_dir, all_voices, save_dir)
    elif args.language:
        voices = _get_voice_names(args.language, languages)
        language = args.language
        if "_" not in args.language:
            if args.language not in languages:
                print("Language", args.language, "not available")
                sys.exit(1)
            language = languages[args.language]
        else:
            if args.language not in URLS:
                print("Language", args.language, "not available")
                sys.exit(1)
            language = languages[args.language]
        for voice in voices:
            process_single(voice, args.output_dir, all_voices, save_dir)
        outdir = Path(args.output_dir)
        langdir = outdir / f"nabucasa_{args.language}"
        if not langdir.is_dir():
            langdir.mkdir()
        merge_voices(voices, outdir, langdir)


if __name__ == "__main__":
    main()
