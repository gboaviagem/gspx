#!/usr/bin/env python
"""Setup to pip package."""

from setuptools import setup
import json
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + "/requirements.txt"
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()
install_requires = list(filter(lambda x: x[:2] != "--", install_requires))

reference = None
try:
    reference = os.environ["DEST_REFERENCE"]
except Exception:
    pass

with open("automation/version.json") as json_file:
    data = json.load(json_file)
    version = data["version"]
    name = data["name"]

setup(
    name=name,
    version=version,
    description="GSP on extension algebras over the real numbers",
    author="Guilherme Boaviagem",
    author_email="guilherme.boaviagem@gmail.com",
    install_requires=install_requires,
    packages=["gspx"],
    package_data={"gspx": [
        "resources/jelly_beans.tiff",
        "resources/uk_towns.tsv",
        "resources/uk_towns_wind.tsv"
    ]},
)
