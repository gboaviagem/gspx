# gspx: Graph Signal Processing on eXtension algebras
Python package for implementing graph signal processing on extension (higher dimensional) algebras (currently, only quaternions)


Higher-dimensional algebras over the real numbers, beyond the complex numbers, present the benefit of encoding many features (dimensions, channels, you name it) within a single element. As such, differently from vector spaces, algebraic operations deal with all these features holistically, at once.

This package aims to implement an extension of [graph signal processing (GSP)](https://arxiv.org/pdf/1712.00468) to higher-dimensional algebras, starting with [quaternions](https://en.wikipedia.org/wiki/Quaternion), aiming towards dealing with signals having elements in a Clifford algebra. This is part of an ongoing doctorate research at the Federal University of Pernambuco (UFPE).

## Instalation

It is possible to pip-install the package from Github,

```sh
pip install git+https://github.com/gboaviagem/gspx@main
```

or clone the repository and run `pip install` through all the requirements.

```sh
git clone https://github.com/gboaviagem/gspx
cd gspx
pip install -r requirements.txt
```
