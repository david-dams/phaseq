# tests for interaction matrix elements
import itertools
import numpy as np
from pyqint import PyQInt, gto, cgf
from phaseq import *

def test_primitive(tolerance = 1e-10):
    """test primitive gaussian interaction"""
    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 0, 10., 0.1, 0.5
    a3, l3, m3, n3, x3, y3, z3 =  0.2, 4, 3, 2, 0.4, 10, 0.9
    a4, l4, m4, n4, x4, y4, z4 =  0.1, 1, 1, 0, 1., 5, 0.3

    # Define the gaussians
    gaussians = [
        [x1, y1, z1, l1, m1, n1, a1],
        [x2, y2, z2, l2, m2, n2, a2],
        [x3, y3, z3, l3, m3, n3, a3],
        [x4, y4, z4, l4, m4, n4, a4],
    ]

    # Initialize the integrator
    integrator = PyQInt()

    gs = jnp.array(gaussians)
    l_max = 2*int(jnp.max(gs[:, 3:6])) + 2
    interaction_jit = jax.jit(lambda g1, g2, g3, g4 : interaction(g1, g2, g3, g4, l_max))

    # Generate all combinations of gaussians
    for g1, g2, g3, g4 in itertools.product(gaussians, repeat=4):
        # Calculate the interaction value
        val = interaction_jit(jnp.array(g1), jnp.array(g2), jnp.array(g3), jnp.array(g4))

        # Create GTO objects
        gto1 = gto(1., [g1[0], g1[1], g1[2]], g1[6], g1[3], g1[4], g1[5])
        gto2 = gto(1., [g2[0], g2[1], g2[2]], g2[6], g2[3], g2[4], g2[5])
        gto3 = gto(1., [g3[0], g3[1], g3[2]], g3[6], g3[3], g3[4], g3[5])
        gto4 = gto(1., [g4[0], g4[1], g4[2]], g4[6], g4[3], g4[4], g4[5])

        # Calculate the reference value
        ref = integrator.repulsion_gto(gto1, gto2, gto3, gto4)

        print(ref, val)
        
        if not np.isclose(ref, val, rtol=tolerance):
            print(f"Values not within tolerance: ref={ref}, val={val}, difference={abs(ref - val)}")
        assert np.isclose(ref, val, rtol=tolerance), f"Values not within tolerance: ref={ref}, val={val}, difference={abs(ref - val)}"
            
def test_contracted(tolerance =  1e-10):
    """test contracted gaussian interactions (i.e. primitive interactions multiplied by coefficients and normalization factors)"""
    pass

def test_derivative(tolerance = 1e-1):    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 0, 10., 0.1, 0.5
    a3, l3, m3, n3, x3, y3, z3 =  0.2, 4, 3, 2, 0.4, 10, 0.9
    a4, l4, m4, n4, x4, y4, z4 =  0.1, 1, 1, 0, 1., 5, 0.3
    
    gaussians = [
        [x1, y1, z1, l1, m1, n1, a1],
        [x2, y2, z2, l2, m2, n2, a2],
        [x3, y3, z3, l3, m3, n3, a3],
        [x4, y4, z4, l4, m4, n4, a4],
    ]
    
    gs = jnp.array(gaussians)
    l_max = 2*int(jnp.max(gs[:, 3:6])) + 2
    func = lambda x : interaction(gs[0], gs[1].at[:3].set(x), gs[2], gs[3], l_max)
    grad = jax.jit(jax.grad(func))
    g = grad( jnp.array([x2, y2, z2]) )

    eps = 1e-8
    num = (func(jnp.array([x2 + eps, y2, z2])) - func(jnp.array([x2, y2, z2]))) / eps
    print(abs(g[0] - num))
    assert abs(g[0] - num) < tolerance

if __name__ == '__main__':
    test_primitive()
    test_derivative()
