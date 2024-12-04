""" from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
import re

import phonemizer
from unidecode import unidecode

# To avoid excessive logging we set the log level of the phonemizer package to Critical
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# Intializing the phonemizer globally significantly reduces the speed
# now the phonemizer is not initialising at every call
# Might be less flexible, but it is much-much faster
global_phonemizers = {}
global_phonemizers["english_cleaners2"] = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Remove brackets
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = global_phonemizers["english_cleaners2"].phonemize([text], strip=True, njobs=1)[0]
    # Added in some cases espeak is not removing brackets
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def polish_cleaners(text):
    text = lowercase(text)
    if not "polish_cleaners" in global_phonemizers:
        global_phonemizers["polish_cleaners"] = phonemizer.backend.EspeakBackend(
            language="pl",
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
            logger=critical_logger,
        )
    phonemes = global_phonemizers["polish_cleaners"].phonemize([text], strip=True, njobs=1)[0]
    # symbols doesn't contain the combining tilde, so replace it with the closest unused character
    # (because the nasal component of Polish "nasal vowels" is 'w', but that's used)
    phonemes = phonemes.replace("\u0303", "ʷ")
    # Added in some cases espeak is not removing brackets
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def hungarian_cleaners(text):
    text = lowercase(text)
    if not "hungarian_cleaners" in global_phonemizers:
        global_phonemizers["hungarian_cleaners"] = phonemizer.backend.EspeakBackend(
            language="hu",
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
            logger=critical_logger,
        )
    phonemes = global_phonemizers["hungarian_cleaners"].phonemize([text], strip=True, njobs=1)[0]
    # Added in some cases espeak is not removing brackets
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def ipa_simplifier(text):
    replacements = [
        ("ɐ", "ə"),
        ("ˈə", "ə"),
        ("ʤ", "dʒ"),
        ("ʧ", "tʃ"),
        ("ᵻ", "ɪ"),
    ]
    for replacement in replacements:
        text = text.replace(replacement[0], replacement[1])
    phonemes = collapse_whitespace(text)
    return phonemes


# I am removing this due to incompatibility with several version of python
# However, if you want to use it, you can uncomment it
# and install piper-phonemize with the following command:
# pip install piper-phonemize

# import piper_phonemize
# def english_cleaners_piper(text):
#     """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
#     text = convert_to_ascii(text)
#     text = lowercase(text)
#     text = expand_abbreviations(text)
#     phonemes = "".join(piper_phonemize.phonemize_espeak(text=text, voice="en-US")[0])
#     phonemes = collapse_whitespace(phonemes)
#     return phonemes
