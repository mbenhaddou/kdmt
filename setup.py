#!/usr/bin/env python

import os.path
from kdmt import version
from setuptools import find_packages, setup

setup(
    name='kdmt',
    version=version.__version__,
    description='Various handy functions and objects',
    long_description=open("README.txt").read() if os.path.isfile("README.txt") else open("README.rst").read(),
    author='Mohamed Ben Haddou',
    author_email='haddomoh@gmail.com',
    url='https://github.com/mbenhaddou/pyutilities',
    packages=find_packages(exclude=["*.tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.6",
)
