# gspx: Graph Signal Processing on eXtension algebras
![coverage](./coverage.svg)

Python package for implementing graph signal processing on extension (higher dimensional) algebras (currently, only quaternions)


Higher-dimensional algebras over the real numbers, beyond the complex numbers, present the benefit of encoding many features (dimensions, channels, you name it) within a single element. As such, differently from vector spaces, algebraic operations deal with all these features holistically, at once.

This package aims to implement an extension of [graph signal processing (GSP)](https://arxiv.org/pdf/1712.00468) to higher-dimensional algebras, starting with [quaternions](https://en.wikipedia.org/wiki/Quaternion), aiming towards dealing with signals having elements in a Clifford algebra. This is part of an ongoing doctorate research at the Federal University of Pernambuco (UFPE).

# Getting started

The tutorial that best demonstrate the use of *quaternion graph signal processing* (QGSP) using `gspx`
is the following:

- [FIR low-pass filter design on quaternion-weighted graph](./notebooks/qlms.ipynb)

The reader may dive deep into more tools in `gspx` through these guided examples:

- [Quaternion matrices](./gspx/utils/README.md)
- [FIR low-pass filter design on *real-weighted* graph](./notebooks/lms.ipynb)

## Instalation

It is recommended to create a separated python environment to run gspx. If one chooses to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (my personal favorite), an appropriate environment is created and open through the lines
```sh
conda create --name gspx_env python=3.7
conda activate gspx_env
```

Then, the packages can be pip-installed from Github,

```sh
python -m pip install git+https://github.com/gboaviagem/gspx@main
```

or one may choose to simply install its dependencies:

```sh
git clone https://github.com/gboaviagem/gspx
cd gspx
bash install.sh
```

## Running unit tests locally

One may run the unit tests by using `pytest`:
```sh
python -m pytest --cov=gspx .
```
To update the coverage badge, run
```sh
rm coverage.svg && coverage-badge -o coverage.svg
```

## Update version in production

Update setup.py version and packages and generate package by running:

```sh
python setup.py sdist bdist_wheel
```

## Acknowledgements

The pre-commit hook used to verify codestyle was copied from
[https://github.com/cbrueffer/pep8-git-hook](https://github.com/cbrueffer/pep8-git-hook).
