from itertools import combinations_with_replacement, product

import jax
import jax.numpy as jnp

from phaseq import *

if __name__ == '__main__':
    
    sto3g = basis_from_json("sto-3g.1.json")
    ch4 = Structure.from_xyz(
        sto3g,
        "ch4.xyz",
        scale = 1.8897259886  # convert from angström to bohr
    )

    res = ch4.scf()
    
    print(ch4.ground_state_energy(res)) # prints something around -40 Ha
    
code = """
mat = [func_interaction(x) for x in ch4.two_electron_arguments]
"""
import timeit
print(timeit.timeit(code, number=1, globals=globals()))

code = """
func_interaction(ch4.two_electron_arguments[10]) 
"""
import timeit
print(timeit.timeit(code, number=100, globals=globals()))


args = jnp.array(ch4.two_electron_arguments)
code = """
mat = jax.vmap()
"""
import timeit
print(timeit.timeit(code, number=1, globals=globals()))


code = """
ref = integrator.repulsion(cgf1, cgf2, cgf3, cgf4)
"""
import timeit
print(timeit.timeit(code, number=1, globals=globals()))

