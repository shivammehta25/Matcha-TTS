import argparse
import random
import sys
import tempfile
from pathlib import Path

import torchaudio
from torch.hub import download_url_to_file

from matcha.utils.data.utils import _extract_zip

INFO = "https://www.openslr.org/83/"

URLS = {
    "irish": {
        "male": "https://www.openslr.org/resources/83/irish_english_male.zip",
    },
    "midlands": {
        "female": "https://www.openslr.org/resources/83/midlands_english_female.zip",
        "male": "https://www.openslr.org/resources/83/midlands_english_male.zip",
    },
    "northern": {
        "female": "https://www.openslr.org/resources/83/northern_english_female.zip",
        "male": "https://www.openslr.org/resources/83/northern_english_male.zip",
    },
    "scottish": {
        "female": "https://www.openslr.org/resources/83/scottish_english_female.zip",
        "male": "https://www.openslr.org/resources/83/scottish_english_male.zip",
    },
    "southern": {
        "female": "https://www.openslr.org/resources/83/southern_english_female.zip",
        "male": "https://www.openslr.org/resources/83/southern_english_male.zip",
    },
    "welsh": {
        "female": "https://www.openslr.org/resources/83/welsh_english_female.zip",
        "male": "https://www.openslr.org/resources/83/welsh_english_male.zip",
    },
}

# Deep breath... "British Isles" is a propaganda term from the Elizabethan conquest.
CITATION = """
  @inproceedings{demirsahin-etal-2020-open,
    title = {{Open-source Multi-speaker Corpora of the English Accents in the British Isles}},
    author = {Demirsahin, Isin and Kjartansson, Oddur and Gutkin, Alexander and Rivera, Clara},
    booktitle = {Proceedings of The 12th Language Resources and Evaluation Conference (LREC)},
    month = may,
    year = {2020},
    pages = {6532--6541},
    address = {Marseille, France},
    publisher = {European Language Resources Association (ELRA)},
    url = {https://www.aclweb.org/anthology/2020.lrec-1.804},
    ISBN = {979-10-95546-34-4},
  }
"""
