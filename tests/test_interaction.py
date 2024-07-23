# tests for nuclear integrals
import numpy as np
from copy import deepcopy
from pyqint import PyQInt, gto

from phaseq import *

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma, factorial

def test_primitive(tolerance = 1e-10):
    """test primitive gaussian nuclear"""
    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 0, 10., 0.1, 0.5
    a3, l3, m3, n3, x3, y3, z3 =  0.2, 4, 3, 2, 0.4, 10, 0.9
    a4, l4, m4, n4, x4, y4, z4 =  0.1, 1, 1, 0, 1., 5, 0.3

    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    gaussian3 = jnp.array( [x3, y3, z3, l3, m3, n3, a3] )
    gaussian4 = jnp.array( [x4, y4, z4, l4, m4, n4, a4] )
        
    val = interaction(gaussian1, gaussian2, gaussian3, gaussian4)
    
    integrator = PyQInt()
    gto1 = gto(1., [x1, y1, z1], a1, l1, m1, n1)
    gto2 = gto(1., [x2, y2, z2], a2, l2, m2, n2)
    gto3 = gto(1., [x3, y3, z3], a3, l3, m3, n3)
    gto4 = gto(1., [x4, y4, z4], a4, l4, m4, n4)

    ref = integrator.interaction_gto(gto1, gto2, gto3, gto4)

    print(abs(ref - val))

    assert abs(ref - val) < tolerance

def test_contracted(tolerance =  1e-10):
    """test contracted gaussian nuclears (i.e. primitive nuclears multiplied by coefficients and normalization factors)"""
    pass

if __name__ == '__main__':
    test_primitive()
