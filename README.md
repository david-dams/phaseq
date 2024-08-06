# PhaseQ 

A JAX-based Hartree-Fock implementation.

## Features

- [x] Restricted Hartree-Fock
- [x] Interfaces with Gaussian basis sets in JSON format as provided by [basissetexchange](https://www.basissetexchange.org/)
- [x] Loads structures from xyz files

## Installation

To install PhaseQ locally, run:

```sh
pip install -e .
```

## Usage

Here's a simple example of how to use PhaseQ:

```python
from phaseq import Structure, basis_from_json

sto3g = basis_from_json("sto3g.json")
ch4 = Structure.from_xyz(
    sto3g,
    "ch4.xyz",
    scale=1.8897259886  # Convert from angstroms to bohrs
)
rho = ch4.scf(tolerance=1e-14)
```

## TODO

- [ ] Clearer testing
- [ ] Improve performance when assembling CGF matrix elements from PGF
- [ ] Implement a post-HF method like MP2 or CI

## Details

The main challenge when implementing Gaussian electron matrix elements in JAX is computing them efficiently in an array-oriented manner. This project achieves that using non-recursive expressions derived by [Takaeta et al.](https://csclub.uwaterloo.ca/~pbarfuss/jpsj.21.2313.pdf). Further details are provided in `deriv.tex`.
