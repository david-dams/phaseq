# tests for nuclear integrals
import numpy as np
from copy import deepcopy
from pyqint import PyQInt, gto

from phaseq import *

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma, factorial

def binomial_ref(a, b):
    if (a < 0) or (b < 0) or (a-b < 0):
        return 1.0
    return factorial(a) / (factorial(b) * factorial(a-b))

def binomial_prefactor_ref(s, l1, l2, x1, x2):
    sum = 0.0
    for t in range(s+1):
        if ((s-l1 <= t) and (t <= l2)):
            sum += (binomial_ref(l2, t) * binomial_ref(l1, s-t) *
                   np.power(x2, l2-t) * np.power(x1, l1-(s-t)))
    return sum

def inner_sum_ref(lim, pax, pbx, px, l1, l2, eps):
    ret = [0.0] * lim
    for i in range(lim):
        for r in range(i//2+1):
            for u in range((i-2*r)//2+1):
                num = (-1)**i * binomial_prefactor_ref(i, l1, l2, pax, pbx) * (-1)**u * factorial(i) * np.pow(px, i - 2*r - 2*u) * np.pow(eps, r+u)
                denom = factorial(r) * factorial(u) * factorial(i - 2*r - 2*u)
                ret[i - 2*r - u] += float(num / denom)
    return ret

def outer_sum_ref(gaussian1, gaussian2, nuc):

    ra, la, aa, rb, lb, ab, g, p, rap, rbp = pack_gaussian_pair(gaussian1, gaussian2)

    l1, m1, n1 = gaussian1[3:6]
    l2, m2, n2 = gaussian2[3:6]

    cp = p - nuc
    px, py, pz = cp
    dcp = jnp.linalg.norm(cp)**2
    d1 = p - gaussian1[:3]
    d2 = p - gaussian2[:3]
    pax, pay, paz = d1
    pbx, pby, pbz = d2    

    limx = int(l1 + l2) + 1
    gx = inner_sum_ref(limx, pax, pbx, px, l1, l2, 1/(4*g))
    limy = int(m1 + m2) + 1
    gy = inner_sum_ref(limy, pay, pby, py, m1, m2, 1/(4*g)) 
    limz = int(n1 + n2) + 1
    gz = inner_sum_ref(limz, paz, pbz, pz, n1, n2, 1/(4*g))

    print(gx, gy, gz)
    

    ret = 0.0    
    for I in range(limx):
        for J in range(limy):
            for K in range(limz):
                ret += boys_fun(I+J+K, dcp * g) * gx[I] * gy[J] * gz[K]

    prefac = -2*np.pi / g * np.exp(-aa*ab*np.linalg.norm(ra-rb)**2/g)
    return prefac * ret, [gx, gy, gz]


def test_sums( tolerance = 1e-10 ):
    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 0, 10., 0.1, 0.5
    nx, ny, nz = 10., 2, 3.
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    nuc = jnp.array([nx, ny, nz])
    
    l_max = max(l1, m1, n1, l2, m2, n2) + 1
    t_arr = jnp.arange(2*l_max+2)

    
    ref11, ref_arr11 = outer_sum_ref(gaussian1, gaussian1, nuc)
    ref12, ref_arr12 = outer_sum_ref(gaussian1, gaussian2, nuc)
    ref22, ref_arr22 = outer_sum_ref(gaussian2, gaussian2, nuc)
    
    integrator = PyQInt()

    gto1 = gto(1., [x1, y1, z1], a1, l1, m1, n1)
    gto2 = gto(1., [x2, y2, z2], a2, l2, m2, n2)

    val11, arr11 = nuclear(gaussian1, gaussian1, nuc, t_arr)
    val12, arr12 = nuclear(gaussian1, gaussian2, nuc, t_arr)
    val22, arr22 = nuclear(gaussian2, gaussian2, nuc, t_arr)

    import pdb; pdb.set_trace()


    print(ref11, val11, abs(ref11 - val11))
    print(ref12, val12, abs(ref12 - val12))
    print(ref22, val22, abs(ref22 - val22))

def test_ref( tolerance = 1e-10 ):
    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 0, 10., 0.1, 0.5
    nx, ny, nz = 10., 2, 3.
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    nuc = jnp.array([nx, ny, nz])
    
    val11,_ = outer_sum_ref(gaussian1, gaussian1, nuc)
    val12,_ = outer_sum_ref(gaussian1, gaussian2, nuc)
    val22,_ = outer_sum_ref(gaussian2, gaussian2, nuc)
    
    integrator = PyQInt()

    gto1 = gto(1., [x1, y1, z1], a1, l1, m1, n1)
    gto2 = gto(1., [x2, y2, z2], a2, l2, m2, n2)

    ref11 = integrator.nuclear_gto(gto1, gto1, nuc)
    ref12 = integrator.nuclear_gto(gto1, gto2, nuc)
    ref22 = integrator.nuclear_gto(gto2, gto2, nuc)

    print(ref11, val11, abs(ref11 - val11))
    print(ref12, val12, abs(ref12 - val12))
    print(ref22, val22, abs(ref22 - val22))

def nuclear_c_matrix(I_range, p, eps, L_range):
    t1 = jnp.pow(-eps, L_range - I_range[:, None])
    t2 = factorial(2*I_range[:, None] - L_range) * factorial(-I_range[:, None] + L_range)
    t3 = jnp.pow(p[:, None, None], 2*I_range[:, None] - L_range)
    return t1 * jnp.nan_to_num(t3/t2, posinf = 0, neginf = 0) * (I_range[:, None] >= L_range//2)

def nuclear(gaussian1, gaussian2, nuc):
    """nuclear potential term (single nucleus) between primitive gaussians.
    
    Args:    
        gaussian1, gaussian2 : array representations of primitive gaussians
        nuc : 3 array, nucleus position

    Returns:
        float, overlap    
    """    
    # pack gaussians
    ra, la, aa, rb, lb, ab, g, p, rap, rbp = pack_gaussian_pair(gaussian1, gaussian2)    
    eps = 1/(4*g)
    
    # relative position of center and nucleus    
    cp = p - nuc
    rcp2 = jnp.pow(jnp.linalg.norm(cp), 2)

    # components of big vector
    l_max = jnp.max(gaussian1[3:6] + gaussian2[3:6])
    l_range = jnp.arange(l_max + 1) # size is l1 + l2 + 1
    pa, pb = p - gaussian1[:3], p - gaussian2[:3]
    bf1 = binomial_prefactor(l_range, gaussian1.at[:3].set(pa), gaussian2.at[:3].set(pb), l_range)
    iterm = (factorial(l_range) * jnp.pow(-1, l_range))[:, None] * bf1
    rterm = jnp.pow(eps, l_range) / factorial(l_range)

    # components of big matrix
    c = nuclear_c_matrix(l_range, cp, eps, l_range)

    # convolution
    As = []
    
    for i in range(3):
        lim = int(gaussian1[3:6][i] + gaussian2[3:6][i]) + 1
        v = iterm[:lim, i]
        a = kernel_to_matrix(rterm, v.size) @ v

        res = c[i, :a.size, :a.size] @ a
        As.append(res)

    import pdb; pdb.set_trace()

    conv = jnp.convolve(jnp.convolve(As[0], As[1]), As[2])

    # function array over unique range
    f_arr = boys_fun( jnp.arange(conv.size), rcp2 * g )

    # raw sum
    res = f_arr @ conv
    
    rab2 = jnp.pow(jnp.linalg.norm(rb - ra), 2)
    prefac = 2*jnp.pi/g * jnp.exp(-aa*ab*rab2/g)
    
    return -prefac * res

def test_primitive(tolerance = 1e-10):
    """test primitive gaussian nuclear"""
    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5
    nx, ny, nz = 1, 2, 3.
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    nuc = jnp.array([nx, ny, nz])
    
    l_max = max(l1, m1, n1, l2, m2, n2) + 1
    t_arr = jnp.arange(2*l_max+2)   
    
    val11 = nuclear(gaussian1, gaussian1, nuc, t_arr)
    val12 = nuclear(gaussian1, gaussian2, nuc, t_arr)
    val22 = nuclear(gaussian2, gaussian2, nuc, t_arr)
    
    integrator = PyQInt()

    gto1 = gto(1., [x1, y1, z1], a1, l1, m1, n1)
    gto2 = gto(1., [x2, y2, z2], a2, l2, m2, n2)

    ref11 = integrator.nuclear_gto(gto1, gto1, nuc)
    ref12 = integrator.nuclear_gto(gto1, gto2, nuc)
    ref22 = integrator.nuclear_gto(gto2, gto2, nuc)

    import pdb; pdb.set_trace()

    
    print(abs(ref11 - val11))
    print(abs(ref12 - val12))
    print(abs(ref22 - val22))

    assert abs(ref11 - val11) < tolerance
    assert abs(ref12 - val12) < tolerance
    assert abs(ref22 - val22) < tolerance

def test_contracted(tolerance =  1e-10):
    """test contracted gaussian nuclears (i.e. primitive nuclears multiplied by coefficients and normalization factors)"""
    pass

if __name__ == '__main__':
    # test_primitive()    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 2, 10., 0.1, 0.5
    nx, ny, nz = 1, 2, 3.
    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
    nuc = jnp.array([nx, ny, nz])

    ref = outer_sum_ref(gaussian1, gaussian2, nuc)
    val = nuclear(gaussian1, gaussian2, nuc)
