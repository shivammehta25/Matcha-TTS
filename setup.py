#!/usr/bin/env python
import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

exts = [
    Extension(
        name="matcha.utils.monotonic_align.core",
        sources=["matcha/utils/monotonic_align/core.pyx"],
    )
]

with open("README.md", "r", encoding="utf-8") as readme_file:
    README = readme_file.read()


setup(
    name="matcha-tts",
    version="0.0.0.1.dev0",
    description="🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Shivam Mehta",
    author_email="shivam.mehta25@gmail.com",
    url="https://shivammehta25.github.io/Matcha-TTS",
    install_requires=[
        str(r)
        for r in open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    ],
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests/*", "examples", "examples/*"]),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "matcha-data-stats=matcha.utils.generate_data_statistics:main",
            "matcha-tts=matcha.cli:cli",
            "matcha-tts-app=matcha.app:main",
        ]
    },
    ext_modules=cythonize(exts, language_level=3),
    python_requires=">=3.9.0",

)
