import jax
import jax.numpy as jnp

from pyqint import PyQInt, cgf, gto
import numpy as np
import math

class Reference:

    @staticmethod
    def binomial_prefactor(s, ia, ib, xpa, xpb):
        from scipy.special import binom
        sum = 0.0
        for t in range(s + 1):
            if (s - ia <= t) and (t <= ib):
                sum += binom(ia, s - t) * binom(ib, t) * (xpa ** (ia - s + t)) * (xpb ** (ib - t))
        return sum

    @staticmethod
    def nuclear(a, l1, m1, n1, alpha1, b, l2, m2, n2, alpha2, c):
        import scipy.special
        
        Fgamma = lambda s, x : scipy.special.gammainc(s+0.5, x) * 0.5 * jnp.pow(x,-s-0.5) * scipy.special.gamma(s+0.5)

        gamma = alpha1 + alpha2

        p = (alpha1 * a + alpha2 * b) / gamma
        rab2 = jnp.linalg.norm(a - b)**2
        rcp2 = jnp.linalg.norm(c - p)**2

        ax = Reference.a_array(l1, l2, p[0] - a[0], p[0] - b[0], p[0] - c[0], gamma)
        ay = Reference.a_array(m1, m2, p[1] - a[1], p[1] - b[1], p[1] - c[1], gamma)
        az = Reference.a_array(n1, n2, p[2] - a[2], p[2] - b[2], p[2] - c[2], gamma)

        sum = 0.0

        for i in range(l1 + l2 + 1):
            for j in range(m1 + m2 + 1):
                for k in range(n1 + n2 + 1):
                    sum += ax[i] * ay[j] * az[k] * Fgamma(i + j + k, rcp2 * gamma)

        return -2.0 * jnp.pi / gamma * jnp.exp(-alpha1 * alpha2 * rab2 / gamma) * sum

    @staticmethod
    def a_array(l1, l2, pa, pb, cp, g):
        imax = l1 + l2 + 1
        arrA = [0] * imax

        for i in range(imax):
            for r in range(int(i/2)+1):
                for u in range( int((i-2*r)/2)+1):
                    iI = i - 2*r - u
                    arrA[iI] += Reference.A_term(i, r, u, l1, l2, pa, pb, cp, g) # some pure function call

        return arrA
    
    @staticmethod
    def A_term(i, r, u, l1, l2, pax, pbx, cpx, gamma):
        return (math.pow(-1, i) * Reference.binomial_prefactor(i, l1, l2, pax, pbx) *
                math.pow(-1, u) * factorial(i) * math.pow(cpx, i - 2 * r - 2 * u) *
                math.pow(0.25 / gamma, r + u) / factorial(r) / factorial(u) / factorial(i - 2 * r - 2 * u))

### QUICK TESTS ###
def test_binom():
    assert False
    
def test_gaussian_norm():
    assert False

def test_double_factorial():
    assert False

def test_binomial_prefactor():
    from tests import Reference

    # primitive gaussian is always [pos, lmn, alpha]
    g1 = jnp.array( [-0.1, 0.3, 0.7, 2, 1, 3, 0.2] )
    g2 = jnp.array( [0.1, 0.4, 0.1, 2, 0, 5, 0.1] )

    # array of combined angular momenta
    l_max = (g1[3:6].max() + g2[3:6].max()) + 1
    l_arr = jnp.arange(l_max)
    t_arr = jnp.arange(2*l_arr.max() + 1)

    bf = binomial_prefactor(2*l_arr, g1, g2, t_arr)

    for comp, arr in enumerate(bf):
        for i in range(3):
            ref = Reference.binomial_prefactor(2*l_arr[comp].astype(int), g1[3+i], g2[3+i], g1[i], g2[i])
            assert jnp.abs(arr[i] - ref) < 1e-10

def test_prim_overlap():
    assert False

def test_prim_kinetic():
    assert False

def test_gto_overlap():
    integrator = PyQInt()

    # parameters
    c_1, c_2 = 0.391957, 0.391957
    alpha_1, alpha_2 = 0.22229, 0.22229
    alpha_1, alpha_2 = 0.3, 0.1
    lmn1, lmn2 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ])
    p_1, p_2 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.])

    # pyqint gtos
    gto_1 = gto(c_1, p_1.tolist(), alpha_1, *(lmn1.tolist()))
    gto_2 = gto(c_2, p_2.tolist(), alpha_2, *(lmn2.tolist()))

    # overlap function
    overlap = get_overlap(jnp.concatenate([lmn1, lmn2]).max()+10)
    overlap = jax.jit(overlap)

    print(integrator.overlap_gto(gto_1, gto_2))
    print(overlap(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2))

    assert abs(integrator.overlap_gto(gto_1, gto_2) - overlap(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2)) < 1e-10

def test_gto_kinetic():
    integrator = PyQInt()

    # parameters
    c_1, c_2 = 0.391957, 0.391957
    alpha_1, alpha_2 = 0.22229, 0.22229
    alpha_1, alpha_2 = 0.3, 0.1
    lmn1, lmn2 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ])
    p_1, p_2 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.])

    # pyqint gtos
    gto_1 = gto(c_1, p_1.tolist(), alpha_1, *(lmn1.tolist()))
    gto_2 = gto(c_2, p_2.tolist(), alpha_2, *(lmn2.tolist()))

    # overlap function
    kinetic = get_kinetic(jnp.concatenate([lmn1, lmn2]).max())
    kinetic = jax.jit(kinetic)

    print(integrator.kinetic_gto(gto_1, gto_2))
    print(kinetic(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2))

    assert abs(integrator.kinetic_gto(gto_1, gto_2) - kinetic(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2)) < 1e-10

def test_a_array():
    l1, l2, pa, pb, cp, g = 2,3,0.1, 0.2, 0.3, 0.1

    a_array = get_a_array(max(l1,l2))
    # jax.make_jaxpr(a_array)(l1, l2, pa, pb, cp, g)
    a_array = jax.jit(a_array)

    print(a_array(l1, l2, pa, pb, cp, g))
    print(Reference.a_array(l1, l2, pa, pb, cp, g))

    assert jnp.allclose( a_array(l1, l2, pa, pb, cp, g)[:(l1+l2+1)], jnp.array(Reference.a_array(l1, l2, pa, pb, cp, g)) )

def test_a_array_grad():
    l1, l2, pa, pb, cp, g = 2,3,0.1, 0.2, 0.3, 0.1
    a_array = get_a_array(max(l1,l2))
    grad = jax.jit(jax.grad( lambda *xs : a_array(*xs).sum(), argnums = [2,3,4,5]))
    grad(l1,l2,pa,pb,cp,g)
    arr = grad(l1,l2,pa,pb,cp,g)    

    assert not jnp.isnan(jnp.array(arr)).any()

def test_b_array_grad():
    l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta = 1, 2, 3, 4, 0.1, 0.2, 0.3, 0.1, 0.1, 0.2, 0.3, 0.1, 0.3

    b_array = get_b_array(max(l1,l2,l3,l4))
    grad = jax.jit(jax.grad( lambda *xs : b_array(*xs).sum(), argnums = [5,6,7,8,9,10,11,12]))
    grad(l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta)
    
    arr = grad(l1, l2, l3, l4, p, a, b, q, c, d, g1, g2, delta)

    assert not jnp.isnan(jnp.array(arr)).any()
    
def test_gto_nuclear():
    integrator = PyQInt()

    # parameters
    c_1, c_2 = 0.391957, 0.391957
    alpha_1, alpha_2 = 0.22229, 0.22229
    alpha_1, alpha_2 = 0.3, 0.1
    lmn1, lmn2 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ])
    p_1, p_2, nuc = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1])

    # pyqint gtos
    gto_1 = gto(c_1, p_1.tolist(), alpha_1, *(lmn1.tolist()))
    gto_2 = gto(c_2, p_2.tolist(), alpha_2, *(lmn2.tolist()))

    # overlap function
    nuclear = get_nuclear(jnp.concatenate([lmn1, lmn2]).max())
    nuclear = jax.jit(nuclear)
    print(nuclear(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, nuc))
    print(integrator.nuclear_gto(gto_1, gto_2, nuc.tolist()))

    assert abs(integrator.nuclear_gto(gto_1, gto_2, nuc.tolist()) - nuclear(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, nuc)) < 1e-10
        
def test_gto_repulsion():
    integrator = PyQInt()

    # parameters
    c_1, c_2, c_3, c_4 = 0.391957, 0.391957, 0.391957, 0.391957
    alpha_1, alpha_2 = 0.22229, 0.22229
    alpha_1, alpha_2, alpha_3, alpha_4 = 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])

    lmn1, lmn2, lmn3, lmn4 = jnp.array([1,0,1 ]), jnp.array([0,1,1 ]), jnp.array([1,1,1 ]), jnp.array([1,0,1 ])

    # pyqint gtos
    gto_1 = gto(c_1, p_1.tolist(), alpha_1, *(lmn1.tolist()))
    gto_2 = gto(c_2, p_2.tolist(), alpha_2, *(lmn2.tolist()))
    gto_3 = gto(c_3, p_3.tolist(), alpha_3, *(lmn3.tolist()))
    gto_4 = gto(c_4, p_4.tolist(), alpha_4, *(lmn4.tolist()))

    # overlap function
    repulsion = get_repulsion(jnp.concatenate([lmn1, lmn2, lmn3, lmn4]).max())
    repulsion = jax.jit(repulsion)
    print(repulsion(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, alpha_3, lmn3, p_3, alpha_4, lmn4, p_4))
    print(integrator.repulsion_gto(gto_1, gto_2, gto_3, gto_4))
    rep = lambda a : repulsion(a, lmn1, p_1, alpha_2, lmn2, p_2, alpha_3, lmn3, p_3, alpha_4, lmn4, p_4) 
    rep_jit = jax.jit(jax.vmap( lambda a : rep(a)))
    
    import timeit
    def foo(): repulsion(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, alpha_3, lmn3, p_3, alpha_4, lmn4, p_4)
    def bar(): integrator.repulsion_gto(gto_1, gto_2, gto_3, gto_4)
    def frisbee() : rep_jit(jnp.arange(1000))

    
    import pdb; pdb.set_trace()

    timeit.timeit(foo, number = 100)
    timeit.timeit(bar, number = 100)
    timeit.timeit(frisbee, number = 1)
    
    import pdb; pdb.set_trace()

    assert abs(repulsion(alpha_1, lmn1, p_1, alpha_2, lmn2, p_2, alpha_3, lmn3, p_3, alpha_4, lmn4, p_4) - integrator.repulsion_gto(gto_1, gto_2, gto_3, gto_4)) < 1e-10
    
def test_overlap():
    
    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 0.391957, 0.4, 0.5, 0.6, 0.391957, 0.4, 0.5, 0.6
    alpha_11, alpha_12, alpha_21, alpha_22,  alpha_31, alpha_32, alpha_41, alpha_42 = 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])


    orb1 = jnp.concatenate( [jnp.array([c_11, c_12]), jnp.array([alpha_11, alpha_12]), lmn1, p_1] )
    orb2 = jnp.concatenate( [jnp.array([c_21, c_22]), jnp.array([alpha_21, alpha_22]), lmn2, p_2] )

    
    cgf1 = cgf(p_1.tolist())
    cgf1.add_gto( c_11, alpha_11, *lmn1.tolist() )
    cgf1.add_gto( c_12, alpha_12, *lmn1.tolist() )

    cgf2 = cgf(p_2.tolist())
    cgf2.add_gto( c_21, alpha_21, *lmn2.tolist() )
    cgf2.add_gto( c_22, alpha_22, *lmn2.tolist() )

    integrator = PyQInt()
    n1 = integrator.overlap(cgf1, cgf2)

    overlap = one_body_wrapper(get_overlap(jnp.concatenate([lmn1, lmn2]).max()), 2)
    n2 = overlap( orb1, orb2 )

    assert abs(n1-n2) < 1e-10
    
def test_kinetic():
    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 0.391957, 0.4, 0.5, 0.6, 0.391957, 0.4, 0.5, 0.6
    alpha_11, alpha_12, alpha_21, alpha_22,  alpha_31, alpha_32, alpha_41, alpha_42 = 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])

    orb1 = jnp.concatenate( [jnp.array([c_11, c_12]), jnp.array([alpha_11, alpha_12]), lmn1, p_1] )
    orb2 = jnp.concatenate( [jnp.array([c_21, c_22]), jnp.array([alpha_21, alpha_22]), lmn2, p_2] )
    
    cgf1 = cgf(p_1.tolist())
    cgf1.add_gto( c_11, alpha_11, *lmn1.tolist() )
    cgf1.add_gto( c_12, alpha_12, *lmn1.tolist() )

    cgf2 = cgf(p_2.tolist())
    cgf2.add_gto( c_21, alpha_21, *lmn2.tolist() )
    cgf2.add_gto( c_22, alpha_22, *lmn2.tolist() )

    integrator = PyQInt()
    n1 = integrator.kinetic(cgf1, cgf2)

    kinetic = one_body_wrapper(get_kinetic(jnp.concatenate([lmn1, lmn2]).max()), 2)    
    n2 = kinetic( orb1, orb2 )

    assert abs(n1-n2) < 1e-10

def test_nuclear():    
    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 0.391957, 0.4, 0.5, 0.6, 0.391957, 0.4, 0.5, 0.6
    alpha_11, alpha_12, alpha_21, alpha_22,  alpha_31, alpha_32, alpha_41, alpha_42 = 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])

    orb1 = jnp.concatenate( [jnp.array([c_11, c_12]), jnp.array([alpha_11, alpha_12]), lmn1, p_1] )
    orb2 = jnp.concatenate( [jnp.array([c_21, c_22]), jnp.array([alpha_21, alpha_22]), lmn2, p_2] )
    
    cgf1 = cgf(p_1.tolist())
    cgf1.add_gto( c_11, alpha_11, *lmn1.tolist() )
    cgf1.add_gto( c_12, alpha_12, *lmn1.tolist() )

    cgf2 = cgf(p_2.tolist())
    cgf2.add_gto( c_21, alpha_21, *lmn2.tolist() )
    cgf2.add_gto( c_22, alpha_22, *lmn2.tolist() )

    integrator = PyQInt()
    n1 = integrator.nuclear(cgf1, cgf2, p_3.tolist(), 1)

    nuclear = one_body_wrapper(get_nuclear(jnp.concatenate([lmn1, lmn2]).max()), 2)    
    n2 = nuclear( orb1, orb2, p_3 )

    assert abs(n1-n2) < 1e-10

def test_repulsion():
    
    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 0.391957, 0.4, 0.5, 0.6, 0.391957, 0.4, 0.5, 0.6
    alpha_11, alpha_12, alpha_21, alpha_22,  alpha_31, alpha_32, alpha_41, alpha_42 = 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.5
    lmn1, lmn2, lmn3, lmn4 = jnp.array([2,0,1 ]), jnp.array([0,3,1 ]), jnp.array([2,2,1 ]), jnp.array([1,0,1 ])
    p_1, p_2, p_3, p_4 = jnp.array([3., 1., 0.]), jnp.array([0, 0, 2.]), jnp.array([1,1,1.]), jnp.array([1.,3,1])


    c_11, c_12, c_21, c_22, c_31, c_32, c_41, c_42 = 1, 1, 1, 1, 1, 1, 1, 1

    orb1 = jnp.concatenate( [jnp.array([c_11, c_12]), jnp.array([alpha_11, alpha_12]), lmn1, p_1] )
    orb2 = jnp.concatenate( [jnp.array([c_21, c_22]), jnp.array([alpha_21, alpha_22]), lmn2, p_2] )
    orb3 = jnp.concatenate( [jnp.array([c_31, c_32]), jnp.array([alpha_31, alpha_32]), lmn3, p_3] )
    orb4 = jnp.concatenate( [jnp.array([c_41, c_42]), jnp.array([alpha_41, alpha_42]), lmn4, p_4] )
    
    cgf1 = cgf(p_1.tolist())
    cgf1.add_gto( c_11, alpha_11, *lmn1.tolist() )
    cgf1.add_gto( c_12, alpha_12, *lmn1.tolist() )

    cgf2 = cgf(p_2.tolist())
    cgf2.add_gto( c_21, alpha_21, *lmn2.tolist() )
    cgf2.add_gto( c_22, alpha_22, *lmn2.tolist() )

    cgf3 = cgf(p_3.tolist())
    cgf3.add_gto( c_31, alpha_31, *lmn3.tolist() )
    cgf3.add_gto( c_32, alpha_32, *lmn3.tolist() )

    cgf4 = cgf(p_4.tolist())
    cgf4.add_gto( c_41, alpha_41, *lmn4.tolist() )
    cgf2.add_gto( c_42, alpha_42, *lmn4.tolist() )

    
    integrator = PyQInt()
    n1 = integrator.repulsion(cgf1, cgf2, cgf3, cgf4)

    repulsion = two_body_wrapper(get_repulsion(jnp.concatenate([lmn1, lmn2, lmn3, lmn4]).max()), 2)
    # repulsion  = jax.jit(repulsion)
    n2 = repulsion( orb1, orb2, orb3, orb4 )

    import pdb; pdb.set_trace()


    assert abs(n1-n2) < 1e-10
