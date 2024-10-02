import argparse
import tempfile
from pathlib import Path

import torchaudio
from torch.hub import download_url_to_file

# from match.utils.data.utils import _extract_zip

URL = "https://github.com/NabuCasa/voice-datasets"

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
    for lang in URLS:
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
    parser.add_argument("-o", "--output-dir", type=str, default="/tmp/data", help="Place to store the converted data")

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
        if not language in URLS:
            return []
        return URLS[language]
    else:
        if not language in languages:
            return []
        voices = []
        for sublang in languages[language]:
            voices += URLS[sublang].keys()
        return voices


# def convert_zip_contents(filename, outdir):
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         _extract_zip(filename, tmpdirname)
#         for webmfile in Path(tmpdirname).glob("*.webm"):
#             pass


def main():
    args = get_args()

    languages = get_languages()
    if args.list_languages:
        print_languages(languages, True)
        exit(1)

    if args.list_voices:
        voices = _get_voice_names(args.list_voices, languages)
        if voices == []:
            print("Language", args.list_voices, "not available")
            print_languages(languages, True)
            exit(1)
        print(", ".join(voices))

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir()

    if args.voice:
        all_voices = get_per_voice()
        if not args.voice in all_voices:
            print("Voice", args.voice, "not available")
            exit(1)
        url = all_voices[args.voice]
        zipfilename = url.split("/")[-1]
        if save_dir:
            download_url_to_file(all_voices[args.voice], str(save_dir / zipfilename), progress=True)
        else:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as zf:
                download_url_to_file(all_voices[args.voice], zf.name, progress=True)


if __name__ == "__main__":
    print(torchaudio.utils.sox_utils.list_read_formats())
    exit(0)
    main()
