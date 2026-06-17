import argparse
import random
import sys

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

# From https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/datasets/cmu_arctic.py
# MIT licence: https://github.com/LCAV/pyroomacoustics/blob/master/LICENSE
cmu_arctic_speakers = {
    "aew": {"sex": "male", "accent": "US"},
    "ahw": {"sex": "male", "accent": "German"},
    "aup": {"sex": "male", "accent": "Indian"},
    "awb": {"sex": "male", "accent": "Scottish"},
    "axb": {"sex": "female", "accent": "Indian"},
    "bdl": {"sex": "male", "accent": "US"},
    "clb": {"sex": "female", "accent": "US"},
    "eey": {"sex": "female", "accent": "US"},
    "fem": {"sex": "male", "accent": "Irish"},
    "gka": {"sex": "male", "accent": "Indian"},
    "jmk": {"sex": "male", "accent": "Canadian"},
    "ksp": {"sex": "male", "accent": "Indian"},
    "ljm": {"sex": "female", "accent": "US"},
    "lnh": {"sex": "female", "accent": "US"},
    "rms": {"sex": "male", "accent": "US"},
    "rxr": {"sex": "male", "accent": "Dutch"},
    "slp": {"sex": "female", "accent": "Indian"},
    "slt": {"sex": "female", "accent": "US"},
}

BASE_TPL = "http://festvox.org/cmu_arctic/packed/cmu_us_{}_arctic.tar.bz2"


def _list_voices(given=""):
    if given != "":
        print(f"Voice {given} not available")
    print("Available voices are:")
    for spkr in cmu_arctic_speakers:
        details = cmu_arctic_speakers[spkr]
        print(f"{spkr}\t{details['accent']} {details['sex']} voice")


def read_text(voice):
    if voice not in cmu_arctic_speakers:
        return []
    textdata = []
    with open(f"data/cmu_us_{voice}_arctic/etc/txt.done.data") as txtdone:
        for line in txtdone.readlines():
            first_paren = line.find("(")
            first_quote = line.find('"')
            last_quote = line.rfind('"')
            fileid = line[first_paren + 1 : first_quote].strip()
            text = line[first_quote + 1 : last_quote].strip()
            textdata.append(f"data/cmu_us_{voice}_arctic/wav/{fileid}.wav|{text}")
    return textdata


def main(args):
    if args.list:
        _list_voices()
        sys.exit(0)

    if args.voice not in cmu_arctic_speakers:
        _list_voices(args.voice)
        sys.exit(1)

    url = BASE_TPL.format(args.voice)

    print("Downloading and unpacking data")
    print("Using url", url)
    # FIXME: change to use the same download/extract as the others
    # download_uncompress(url, path="./data")
    text = read_text(args.voice)
    random.shuffle(text)

    print("Generating data splits and writing text")
    split_size = int(len(text) / (100 / 5))
    train_sents = text[:split_size]
    test_sents = text[split_size:]
    if args.valid:
        valid_sents = train_sents[:split_size]
        train_sents = train_sents[split_size:]
        with open(f"data/filelists/cmu_us_{args.voice}_arctic_val_filelist.txt", "w") as val:
            val.write("\n".join(valid_sents))
    with open(f"data/filelists/cmu_us_{args.voice}_arctic_train_filelist.txt", "w") as val:
        val.write("\n".join(train_sents))
    with open(f"data/filelists/cmu_us_{args.voice}_arctic_test_filelist.txt", "w") as val:
        val.write("\n".join(test_sents))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--voice", type=str, default="rms", required=True, help="ID of voice to use")
    parser.add_argument("-s", "--split", type=int, default=5, required=False, help="Size of test(/validation) split")
    parser.add_argument(
        "-V", "--valid", action="store_true", default=False, required=False, help="Create a validation set"
    )
    parser.add_argument("-l", "--list", action="store_true", default=False, required=False, help="List speakers")
    args = parser.parse_args()

    main(args)
