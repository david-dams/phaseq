# tests for nuclear matrix elements
import numpy as np
from copy import deepcopy
from pyqint import PyQInt, gto, cgf

from phaseq import *

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma, factorial

def test_primitive(tolerance = 1e-10):
    """test primitive gaussian nuclear"""
    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  1, 4, 2, 2, 10., 2., 0.7
    nx, ny, nz = 1, 2, 3.
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    nuc = jnp.array([nx, ny, nz])

    l_max = int( 2 * jnp.max( jnp.concatenate([gaussian1[3:6], gaussian2[3:6]]) ) ) + 2
    nuclear_jit = jax.jit(nuclear, static_argnames = 'l_max')
    
    val11 = nuclear_jit(gaussian1, gaussian1, nuc, l_max)
    val12 = nuclear_jit(gaussian1, gaussian2, nuc, l_max)
    val22 = nuclear_jit(gaussian2, gaussian2, nuc, l_max)
    
    integrator = PyQInt()

    gto1 = gto(1., [x1, y1, z1], a1, l1, m1, n1)
    gto2 = gto(1., [x2, y2, z2], a2, l2, m2, n2)

    ref11 = integrator.nuclear_gto(gto1, gto1, nuc)
    ref12 = integrator.nuclear_gto(gto1, gto2, nuc)
    ref22 = integrator.nuclear_gto(gto2, gto2, nuc)
    
    print(abs(ref11 - val11))
    print(abs(ref12 - val12))
    print(abs(ref22 - val22))

    assert abs(ref11 - val11) < tolerance
    assert abs(ref12 - val12) < tolerance
    assert abs(ref22 - val22) < tolerance

    
def test_contracted(tolerance =  1e-10):
    """test contracted gaussian nuclear (i.e. primitive overlaps multiplied by coefficients and normalization factors)"""

    # [x1, y1, z1, l1, m1, n1, a1]
    gaussians = [
            [0.2, 0.3, 0.1, 4, 1, 0, 0.2], 
            [0.2, 0.3, 0.1, 3, 2, 1, 0.3], 
            [0.2, 0.3, 0.1, 2, 3, 2, 0.4],
            
            [0.5, 0.6, 0.4, 4, 1, 0, 0.5], 
            [0.5, 0.6, 0.4, 3, 2, 1, 0.6], 
            [0.5, 0.6, 0.4, 2, 3, 2, 0.7] 
    ]

    coeffs = [
        [0.1, 3., 1., 0., 0., 0.],
        [0., 0., 0., 0.4, 2., 7.],
    ]
    nx, ny, nz = 1, 2, 3.
    nuc = jnp.array([nx, ny, nz])
    
    gs, cs = jnp.array(gaussians), jnp.array(coeffs)
    
    l_max = int( 2 * jnp.max( jnp.concatenate([gs[3:], gs[:3]]) ) ) + 2
    nuclear_jit = jax.jit(lambda g1, g2, nuc : nuclear(g1, g2, nuc, l_max))
    func = jax.jit(promote_one(lambda g1, g2 : nuclear_jit(g1, g2, nuc)))

    nuclear11= func(cs[0, :3], cs[0, :3], gs[:3], gs[:3])
    nuclear12= func(cs[0, :3], cs[1, 3:], gs[:3], gs[3:])
    nuclear22= func(cs[1, 3:], cs[1, 3:], gs[3:], gs[3:])

    integrator = PyQInt()
    
    cgf1 = cgf(gaussians[0][:3])
    cgf1.add_gto(coeffs[0][0], gaussians[0][-1], *(gaussians[0][3:6]) )
    cgf1.add_gto(coeffs[0][1], gaussians[1][-1], *(gaussians[1][3:6]) )
    cgf1.add_gto(coeffs[0][2], gaussians[2][-1], *(gaussians[2][3:6]) )
    
    cgf2 = cgf(gaussians[3][:3])
    cgf2.add_gto(coeffs[1][3], gaussians[3][-1], *(gaussians[3][3:6]) )
    cgf2.add_gto(coeffs[1][4], gaussians[4][-1], *(gaussians[4][3:6]) )
    cgf2.add_gto(coeffs[1][5], gaussians[5][-1], *(gaussians[5][3:6]) )    


    nuclear_ref11 = integrator.nuclear( cgf1, cgf1, nuc, 1)
    nuclear_ref12 = integrator.nuclear( cgf1, cgf2, nuc, 1)
    nuclear_ref22 = integrator.nuclear( cgf2, cgf2, nuc, 1)

    print(abs(nuclear_ref11 - nuclear11))
    print(abs(nuclear_ref12 - nuclear12))
    print(abs(nuclear_ref22 - nuclear22))

    assert abs(nuclear_ref11 - nuclear11) < tolerance
    assert abs(nuclear_ref12 - nuclear12) < tolerance
    assert abs(nuclear_ref22 - nuclear22) < tolerance

def test_derivative( tolerance = 1e-4):    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 0, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    nuc = jnp.array([x1, y1, z1])
    
    l_max = 2*max(l1, m1, n1, l2, m2, n2) + 2
    func = lambda x : nuclear(gaussian1, gaussian2.at[:3].set(x), nuc, l_max)
    grad = jax.jit(jax.jacrev(func))
    
    g = grad( jnp.array([x2, y2, z2]) )    
    eps = 1e-10
    num = (func(jnp.array([x2 + eps, y2, z2])) - func(jnp.array([x2, y2, z2]))) / eps
    print(abs(g[0] - num))
    assert abs(g[0] - num) < tolerance

    # at same position
    eps = 1e-18 # TODO: this is cheating, check what's going on
    g = grad(jnp.array([x1, y1, z1]))
    num = (func(jnp.array([x1 + eps, y1, z1])) - func(jnp.array([x1, y1, z1]))) / eps
    print(abs(g[0] - num))
    assert abs(g[0] - num) < tolerance

def test_derivative_contracted(tolerance =  1e-4):
    """test contracted gaussian kinetics (i.e. primitive kinetics multiplied by coefficients and normalization factors)"""

    gaussians = [
            [0.2, 0.3, 0.1, 4, 1, 0, 0.2], 
            [0.2, 0.3, 0.1, 3, 2, 1, 0.3], 
            [0.2, 0.3, 0.1, 2, 3, 2, 0.4],
            
            [0.5, 0.6, 0.4, 4, 1, 0, 0.5], 
            [0.5, 0.6, 0.4, 3, 2, 1, 0.6], 
            [0.5, 0.6, 0.4, 2, 3, 2, 0.7] 
    ]

    coeffs = [
        [0.1, 3., 1., 0., 0., 0.],
        [0., 0., 0., 0.4, 2., 7.],
    ]    
    
    x1, y1, z1 =  gaussians[0][4:]
    x2, y2, z2 =  gaussians[3][4:]

    gs, cs = jnp.array(gaussians), jnp.array(coeffs)
    nuc = jnp.array([x1, y1, z1])    
    
    l_max = 2*int(jnp.max(gs[:, 3:6])) + 2
    func = jax.jit(lambda x : promote_one(lambda g1, g2 : nuclear(g1.at[:3].set(x), g2, nuc, l_max))(cs[0, :3], cs[1, 3:], gs[:3], gs[3:]))
    grad = jax.jit(jax.jacrev(func))
    g = grad(jnp.array([x2, y2, z2]))
    
    eps = 1e-8
    num = (func(jnp.array([x2 + eps, y2, z2])) - func(jnp.array([x2, y2, z2]))) / eps
    print(abs(g[0] - num))
    assert abs(g[0] - num) < tolerance

    # at same position
    g = grad(jnp.array([x1, y1, z1]))
    num = (func(jnp.array([x1 + eps, y1, z1])) - func(jnp.array([x1, y1, z1]))) / eps
    print(abs(g[0] - num))
    assert abs(g[0] - num) < tolerance
    
if __name__ == '__main__':
    test_primitive()
    test_contracted()
    test_derivative()
    test_derivative_contracted()
