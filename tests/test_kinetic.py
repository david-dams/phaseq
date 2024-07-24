# tests for kinetic matrix elements
import numpy as np
from copy import deepcopy
from pyqint import PyQInt, gto, cgf

from phaseq import *

# pyqint : c, alpha, l, m, n, (R in GF),  gto(_c, _p, _alpha, _l, _m, _n)
# phaseq : [pos, lmn, alpha]

def test_primitive(tolerance = 1e-10):
    """test primitive gaussian kinetic"""
    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    
    l_max = max(l1, m1, n1, l2, m2, n2) + 1
    l_arr = jnp.arange(l_max+2)
    t_arr = jnp.arange(2*l_max+2)   
    
    val11 = kinetic(gaussian1, gaussian1, l_arr, t_arr)
    val12 = kinetic(gaussian1, gaussian2, l_arr, t_arr)
    val22 = kinetic(gaussian2, gaussian2, l_arr, t_arr)
    
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
    """test contracted gaussian kinetic (i.e. primitive overlaps multiplied by coefficients and normalization factors)"""

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
    
    l_max = jnp.max(gs[:, 3:6]) + 1
    l_arr = jnp.arange(l_max+2) 
    t_arr = jnp.arange(2*l_max+2)       
    func = promote_one(lambda g1, g2 : kinetic(g1, g2, l_arr, t_arr))

    
    kinetic11= func(cs[0, :3], cs[0, :3], gs[:3], gs[:3])
    kinetic12= func(cs[0, :3], cs[1, 3:], gs[:3], gs[3:])
    kinetic22= func(cs[1, 3:], cs[1, 3:], gs[3:], gs[3:])

    integrator = PyQInt()
    
    cgf1 = cgf(gaussians[0][:3])
    cgf1.add_gto(coeffs[0][0], gaussians[0][-1], *(gaussians[0][3:6]) )
    cgf1.add_gto(coeffs[0][1], gaussians[1][-1], *(gaussians[1][3:6]) )
    cgf1.add_gto(coeffs[0][2], gaussians[2][-1], *(gaussians[2][3:6]) )
    
    cgf2 = cgf(gaussians[3][:3])
    cgf2.add_gto(coeffs[1][3], gaussians[3][-1], *(gaussians[3][3:6]) )
    cgf2.add_gto(coeffs[1][4], gaussians[4][-1], *(gaussians[4][3:6]) )
    cgf2.add_gto(coeffs[1][5], gaussians[5][-1], *(gaussians[5][3:6]) )    


    kinetic_ref11 = integrator.kinetic( cgf1, cgf1 )
    kinetic_ref12 = integrator.kinetic( cgf1, cgf2 )
    kinetic_ref22 = integrator.kinetic( cgf2, cgf2 )

    print(abs(kinetic_ref11 - kinetic11))
    print(abs(kinetic_ref12 - kinetic12))
    print(abs(kinetic_ref22 - kinetic22))

    assert abs(kinetic_ref11 - kinetic11) < tolerance
    assert abs(kinetic_ref12 - kinetic12) < tolerance
    assert abs(kinetic_ref22 - kinetic22) < tolerance

def test_derivative():    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 0, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    
    l_max = max(l1, m1, n1, l2, m2, n2) + 1
    l_arr = jnp.arange(l_max)
    t_arr = jnp.arange(2*l_max+1)   

    func = lambda x : kinetic(gaussian1, gaussian2.at[:3].set(x), l_arr, t_arr)
    grad = jax.jit(jax.jacrev(func))
    g = grad( jnp.array([x2, y2, z2]) )

def test_jit():    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 0, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    
    l_max = max(l1, m1, n1, l2, m2, n2) + 1
    l_arr = jnp.arange(l_max+2)
    t_arr = jnp.arange(2*l_max+2)       
    func = jax.jit(lambda g1, g2 : kinetic(g1, g2, l_arr, t_arr))    
    func(gaussian1, gaussian2)
    func(gaussian1, gaussian2)

if __name__ == '__main__':
    test_contracted()
