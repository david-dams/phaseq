# tests for overlap matrix elements
import numpy as np
from copy import deepcopy
from pyqint import PyQInt, gto, cgf

from phaseq import *

# pyqint : c, alpha, l, m, n, (R in GF),  gto(_c, _p, _alpha, _l, _m, _n)
# phaseq : [pos, lmn, alpha]
    
def test_primitive(tolerance = 1e-10):
    """test primitive gaussian overlaps"""
    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 2, 1, 0, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5

    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 0, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    
    l_max = max(l1, m1, n1, l2, m2, n2) + 1
    overlap_jit = jax.jit(overlap, static_argnames = ['l_max'])
    
    val11 = overlap_jit(gaussian1, gaussian1, l_max)
    val12 = overlap_jit(gaussian1, gaussian2, l_max)
    val22 = overlap_jit(gaussian2, gaussian2, l_max)
    
    integrator = PyQInt()

    gto1 = gto(1., [x1, y1, z1], a1, l1, m1, n1)
    gto2 = gto(1., [x2, y2, z2], a2, l2, m2, n2)

    ref11 = integrator.overlap_gto(gto1, gto1)
    ref12 = integrator.overlap_gto(gto1, gto2)
    ref22 = integrator.overlap_gto(gto2, gto2)

    print(abs(ref11 - val11))
    print(abs(ref12 - val12))
    print(abs(ref22 - val22))

    assert abs(ref11 - val11) < tolerance
    assert abs(ref12 - val12) < tolerance
    assert abs(ref22 - val22) < tolerance

def test_contracted(tolerance =  1e-10):
    """test contracted gaussian overlaps (i.e. primitive overlaps multiplied by coefficients and normalization factors)"""

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
    
    gs, cs = jnp.array(gaussians), jnp.array(coeffs)
    
    l_max = int(jnp.max(gs[:, 3:6]) + 1)
    func = jax.jit(promote_one(lambda g1, g2 : overlap(g1, g2, l_max)))

    overlap11= func(cs[0, :3], cs[0, :3], gs[:3], gs[:3])    
    overlap12= func(cs[0, :3], cs[1, 3:], gs[:3], gs[3:])
    overlap22= func(cs[1, 3:], cs[1, 3:], gs[3:], gs[3:])    

    integrator = PyQInt()
    
    cgf1 = cgf(gaussians[0][:3])
    cgf1.add_gto(coeffs[0][0], gaussians[0][-1], *(gaussians[0][3:6]) )
    cgf1.add_gto(coeffs[0][1], gaussians[1][-1], *(gaussians[1][3:6]) )
    cgf1.add_gto(coeffs[0][2], gaussians[2][-1], *(gaussians[2][3:6]) )
    
    cgf2 = cgf(gaussians[3][:3])
    cgf2.add_gto(coeffs[1][3], gaussians[3][-1], *(gaussians[3][3:6]) )
    cgf2.add_gto(coeffs[1][4], gaussians[4][-1], *(gaussians[4][3:6]) )
    cgf2.add_gto(coeffs[1][5], gaussians[5][-1], *(gaussians[5][3:6]) )    

    overlap_ref11 = integrator.overlap( cgf1, cgf1 )
    overlap_ref12 = integrator.overlap( cgf1, cgf2 )
    overlap_ref22 = integrator.overlap( cgf2, cgf2 )
    
    print(abs(overlap_ref11 - overlap11))
    print(abs(overlap_ref12 - overlap12))
    print(abs(overlap_ref22 - overlap22))

    assert abs(overlap_ref11 - overlap11) < tolerance
    assert abs(overlap_ref12 - overlap12) < tolerance
    assert abs(overlap_ref22 - overlap22) < tolerance

def test_derivative( tolerance = 1e-4):    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 0, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    
    l_max = max(l1, m1, n1, l2, m2, n2) + 1
    func = lambda x : overlap(gaussian1, gaussian2.at[:3].set(x), l_max)
    grad = jax.jit(jax.jacrev(func))
    g = grad( jnp.array([x2, y2, z2]) )
    
    eps = 1e-8
    num = (func(jnp.array([x2 + eps, y2, z2])) - func(jnp.array([x2, y2, z2]))) / eps

    print(abs(g[0] - num))
    assert abs(g[0] - num) < tolerance


def test_derivative_contracted(tolerance =  1e-4):
    """test contracted gaussian overlaps (i.e. primitive overlaps multiplied by coefficients and normalization factors)"""

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
    
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5

    gs, cs = jnp.array(gaussians), jnp.array(coeffs)
    
    l_max = int(jnp.max(gs[:, 3:6]) + 1)
    func = jax.jit(lambda x : promote_one(lambda g1, g2 : overlap(g1.at[:3].set(x), g2, l_max))(cs[0, :3], cs[1, 3:], gs[:3], gs[3:]))
    grad = jax.jit(jax.jacrev(func))    
    g = grad(jnp.array([x2, y2, z2]))

    eps = 1e-8
    num = (func(jnp.array([x2 + eps, y2, z2])) - func(jnp.array([x2, y2, z2]))) / eps
    
    print(abs(g[0] - num))
    assert abs(g[0] - num) < tolerance
    
if __name__ == '__main__':
    test_primitive()
    test_contracted()
    test_derivative()
    test_derivative_contracted()
    
