# tests for kinetic integrals
import numpy as np
from copy import deepcopy
from pyqint import PyQInt, gto

from phaseq import *

# pyqint : c, alpha, l, m, n, (R in GF),  gto(_c, _p, _alpha, _l, _m, _n)
# phaseq : [pos, lmn, alpha]

def test_primitive(tolerance = 1e-10):
    """test primitive gaussian kinetics"""
    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 0, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    
    l_max = max(l1, m1, n1, l2, m2, n2) + 1
    l_arr = jnp.arange(l_max+2)
    t_arr = jnp.arange(2*l_max+2)   
    
    val11 = kinetic(l_arr, gaussian1, gaussian1, t_arr)
    val12 = kinetic(l_arr, gaussian1, gaussian2, t_arr)
    val22 = kinetic(l_arr, gaussian2, gaussian2, t_arr)
    
    integrator = PyQInt()

    gto1 = gto(1., [x1, y1, z1], a1, l1, m1, n1)
    gto2 = gto(1., [x2, y2, z2], a2, l2, m2, n2)

    ref11 = integrator.kinetic_gto(gto1, gto1)
    ref12 = integrator.kinetic_gto(gto1, gto2)
    ref22 = integrator.kinetic_gto(gto2, gto2)

    print(abs(ref11 - val11))
    print(abs(ref12 - val12))
    print(abs(ref22 - val22))

    assert abs(ref11 - val11) < tolerance
    assert abs(ref12 - val12) < tolerance
    assert abs(ref22 - val22) < tolerance

def test_contracted(tolerance =  1e-10):
    """test contracted gaussian kinetics (i.e. primitive kinetics multiplied by coefficients and normalization factors)"""
    pass

if __name__ == '__main__':
    test_primitive()
