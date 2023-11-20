#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="hf-fastup",
    description="Fast upload in parallel large datasets to HuggingFace Datasets hub.",
    author="Khaled Koutini",
    author_email="first.last@jku.at",
    url="https://github.com/kkoutini/hf-fastup",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/kkoutini/hf-fastup/issues",
        "Source Code": "https://github.com/kkoutini/hf-fastup",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=["datasets>=2.15.0", "hf_transfer>=0.1.4"],
)
