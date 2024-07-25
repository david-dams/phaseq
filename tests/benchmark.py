import timeit
import numpy as np
from pyqint import PyQInt, gto, cgf
from phaseq import *

if __name__ == '__main__':

    gaussians = [
            [0.2, 0.3, 0.1, 4, 1, 0, 0.2], 
            [0.2, 0.3, 0.1, 3, 2, 1, 0.3], 
            [0.2, 0.3, 0.1, 2, 3, 2, 0.4],

            [0.5, 0.6, 0.4, 4, 1, 0, 0.5], 
            [0.5, 0.6, 0.4, 3, 2, 1, 0.6], 
            [0.5, 0.6, 0.4, 2, 3, 2, 0.7] 
    ]

    coeffs = [
        [0.1, 3., 1.,],
        [0.4, 2., 7.],
        [0.3, 4., 0.1,],
        [0.7, 1., 2.],
    ]    

    gs, cs = jnp.array(gaussians), jnp.array(coeffs)

    l_max = 2*int(jnp.max(gs[:, 3:6])) + 2

    func = jax.jit(promote_two(lambda g1, g2, g3, g4 : interaction(g1, g2, g3, g4, l_max)))

    val = func(cs[0], cs[1], cs[2], cs[3], gs[:3], gs[3:], gs[:3], gs[3:])    

    integrator = PyQInt()

    cgf1 = cgf(gaussians[0][:3])
    cgf1.add_gto(coeffs[0][0], gaussians[0][-1], *(gaussians[0][3:6]) )
    cgf1.add_gto(coeffs[0][1], gaussians[1][-1], *(gaussians[1][3:6]) )
    cgf1.add_gto(coeffs[0][2], gaussians[2][-1], *(gaussians[2][3:6]) )

    cgf2 = cgf(gaussians[3][:3])
    cgf2.add_gto(coeffs[1][0], gaussians[3][-1], *(gaussians[3][3:6]) )
    cgf2.add_gto(coeffs[1][1], gaussians[4][-1], *(gaussians[4][3:6]) )
    cgf2.add_gto(coeffs[1][2], gaussians[5][-1], *(gaussians[5][3:6]) )    

    cgf3 = cgf(gaussians[0][:3])
    cgf3.add_gto(coeffs[2][0], gaussians[0][-1], *(gaussians[0][3:6]) )
    cgf3.add_gto(coeffs[2][1], gaussians[1][-1], *(gaussians[1][3:6]) )
    cgf3.add_gto(coeffs[2][2], gaussians[2][-1], *(gaussians[2][3:6]) )

    cgf4 = cgf(gaussians[3][:3])
    cgf4.add_gto(coeffs[3][0], gaussians[3][-1], *(gaussians[3][3:6]) )
    cgf4.add_gto(coeffs[3][1], gaussians[4][-1], *(gaussians[4][3:6]) )
    cgf4.add_gto(coeffs[3][2], gaussians[5][-1], *(gaussians[5][3:6]) )    


    # Define the first snippet as a function
    def snippet1():
        integrator.repulsion(cgf1, cgf2, cgf3, cgf4)

    # Define the second snippet as a function
    def snippet2():
        func(cs[0], cs[1], cs[2], cs[3], gs[:3], gs[3:], gs[:3], gs[3:])

    # Use timeit to measure the execution time of each snippet
    time_snippet1 = timeit.timeit(snippet1, number=100)
    time_snippet2 = timeit.timeit(snippet2, number=100)

    print(f"Time for snippet 1: {time_snippet1} seconds")
    print(f"Time for snippet 2: {time_snippet2} seconds")
