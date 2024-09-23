""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
phonemes = [
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
    "pau",
    "sil",
]

extra_symbols = [
    "^",
    "$", 
    "?",
    "_",
    "#",
    "[",
    "]",
]
_pad = "~"


symbols = [_pad] + list(phonemes) + list(extra_symbols)

# SPACE_ID = symbols.index(" ")
