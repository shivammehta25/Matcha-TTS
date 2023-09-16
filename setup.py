#!/usr/bin/env python
import os

import numpy
import pkg_resources
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

exts = [
    Extension(
        name="matcha.utils.monotonic_align.core",
        sources=["matcha/utils/monotonic_align/core.pyx"],
    )
]

setup(
    name="matcha",
    version="0.1.0",
    description="A fast TTS architecture with conditional flow matching",
    author="Shivam Mehta",
    author_email="shivam.mehta25@gmail.com",
    url="https://shivammehta25.github.io/Matcha-TTS",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(open(os.path.join(os.path.dirname(__file__), "requirements.txt")))
    ],
    include_dirs=[numpy.get_include()],
    packages=find_packages(exclude=["tests", "tests/*", "examples", "examples/*"]),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "matcha-data-stats=matcha.utils.generate_data_statistics:main",
            "matcha_tts=matcha.cli:cli",
            "matcha_tts_app=matcha.app:main",
        ]
    },
    ext_modules=cythonize(exts, language_level=3),
)
