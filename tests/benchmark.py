import timeit
import numpy as np
from pyqint import PyQInt, gto, cgf
from phaseq import *

if __name__ == '__main__':

    # [c1, x1, y1, z1, l1, m1, n1, a1]
    cgfs = [
        [0.1, 0.2, 0.3, 0.1, 4, 1, 0, 0.2], 
        [3., 0.2, 0.3, 0.1, 3, 2, 1, 0.3], 
        [1., 0.2, 0.3, 0.1, 2, 3, 2, 0.4],
        
        [0.4, 0.5, 0.6, 0.4, 4, 1, 0, 0.5], 
        [2., 0.5, 0.6, 0.4, 3, 2, 1, 0.6], 
        [7., 0.5, 0.6, 0.4, 2, 3, 2, 0.7],
        
        [0.3, 0.2, 0.3, 0.1, 4, 1, 0, 0.2], 
        [4., 0.2, 0.3, 0.1, 3, 2, 1, 0.3], 
        [0.1, 0.2, 0.3, 0.1, 2, 3, 2, 0.4],
        
        [0.7, 0.5, 0.6, 0.4, 4, 1, 0, 0.5], 
        [1., 0.5, 0.6, 0.4, 3, 2, 1, 0.6], 
        [2., 0.5, 0.6, 0.4, 2, 3, 2, 0.7] 

    ]
    
    cgfs = jnp.array(cgfs)
    
    l_max = int(jnp.max(cgfs[:, 4:7])) + 1
    func = matrix_elements(l_max)[3]
    
    c1 = cgfs[:3]
    c2 = cgfs[3:6]
    c3 = cgfs[6:9]
    c4 = cgfs[9:]

    val = func(c1, c2, c3, c4)    

    integrator = PyQInt()
    
    cgf1 = cgf(cgfs[0,1:4])
    cgf1.add_gto(cgfs[0,0], cgfs[0,-1], *(cgfs[0,4:7].astype(int).tolist()) )
    cgf1.add_gto(cgfs[1,0], cgfs[1,-1], *(cgfs[1,4:7].astype(int).tolist()) )
    cgf1.add_gto(cgfs[2,0], cgfs[2,-1], *(cgfs[2,4:7].astype(int).tolist()) )
    
    cgf2 = cgf(cgfs[3,1:4])
    cgf2.add_gto(cgfs[3,0], cgfs[3,-1], *(cgfs[3,4:7].astype(int).tolist()))
    cgf2.add_gto(cgfs[4,0], cgfs[4,-1], *(cgfs[4,4:7].astype(int).tolist()))
    cgf2.add_gto(cgfs[5,0], cgfs[5,-1], *(cgfs[5,4:7].astype(int).tolist()))    

    cgf3 = cgf(cgfs[6,1:4])
    cgf3.add_gto(cgfs[6,0], cgfs[6,-1], *(cgfs[6,4:7].astype(int).tolist()) )
    cgf3.add_gto(cgfs[7,0], cgfs[7,-1], *(cgfs[7,4:7].astype(int).tolist()) )
    cgf3.add_gto(cgfs[8,0], cgfs[8,-1], *(cgfs[8,4:7].astype(int).tolist()) )
    
    cgf4 = cgf(cgfs[9,1:4])
    cgf4.add_gto(cgfs[9,0], cgfs[9,-1], *(cgfs[9,4:7].astype(int).tolist()))
    cgf4.add_gto(cgfs[10,0], cgfs[10,-1], *(cgfs[10,4:7].astype(int).tolist()))
    cgf4.add_gto(cgfs[11,0], cgfs[11,-1], *(cgfs[11,4:7].astype(int).tolist()))    

    # Define the first snippet as a function
    def snippet1():
        integrator.repulsion(cgf1, cgf2, cgf3, cgf4)

    # Define the second snippet as a function
    def snippet2():
        func(c1, c2, c3, c4)
        
    # Use timeit to measure the execution time of each snippet
    time_snippet1 = timeit.timeit(snippet1, number=100)
    time_snippet2 = timeit.timeit(snippet2, number=100)

    print(f"Time for snippet 1: {time_snippet1} seconds")
    print(f"Time for snippet 2: {time_snippet2} seconds")
