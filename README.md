# phaseQ

Hartree-Fock in Python, just-in-time compiled and automatically differentiable, built on JAX.

## Features

- [x] Restricted Hartree-Fock
- [x] Differentiable + JITable
- [x] Supports basis sets specified in MolSSI BSE schema 0.1, as from [basissetexchange](https://www.basissetexchange.org/)
- [x] Supports xyz files

## Installation

To install phaseQ locally, clone this repository and run:

```sh
pip install .
```

## Usage

A basic example to compute the GS energy of methane is

```python
from phaseq import Structure, basis_from_json

# load a basis set
sto3g = basis_from_json("sto-3g.1.json")

# load methane into a "Structure" object
ch4 = Structure.from_xyz(
	sto3g,
	"ch4.xyz",
	scale = 1.8897259886  # convert from angström to bohr
)

# run a restricted Hartree-Fock calculation
res = ch4.scf()
    
# this should be something around -40 Ha	
print(ch4.ground_state_energy(res)) 
```

Right now, autodiff is useful for computing forces or susceptibilities.

## Details

The main challenge when implementing Hartree-Fock in JAX is computing electron matrix elements efficiently in an array-oriented manner with tensors of static shapes, hopefully allowing XLA to fuse many operations.

phaseQ achieves that by relying on transformations of non-recursive expressions derived by [Takaeta et al.](https://csclub.uwaterloo.ca/~pbarfuss/jpsj.21.2313.pdf). Further details are provided in `math.pdf`. 

### Performance

As a consequence of replacing explicit loops with vectorized operations, high angular momentum matrix elements should be quite fast to compute. 

Overall, phaseQ is not that fast on CPU, but achieves reasonable performance if JITed on GPU. YMMV, of course. 

That said, there is plenty of room for relatively easy optimization, e.g.

1. invariants (often involving powers of factorial arrays) are recomputed on every matrix element function call
2. matrix elements are computed from orbital tuples instead of batches 
3. contracted matrices are assembled by just calling into `jnp.einsum` every time
4. so far, there is *no parallelization for cgf matrix elements*.
