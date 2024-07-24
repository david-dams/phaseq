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

def inner_sum_ref_restricted(lim, pax, pbx, px, l1, l2, eps):
    ret = [0.0] * lim
    for i in range(lim):
        for r in range(i//2+1):
            for u in range((i-2*r)//2+1):
                num = (-1)**i * binomial_prefactor_ref(i, l1, l2, pax, pbx) * (-1)**u * factorial(i) * np.pow(px, i - 2*r - 2*u) * np.pow(eps, r+u)
                denom = factorial(r) * factorial(u) * factorial(i - 2*r - 2*u)
                ret[i - 2*r - u] += float(num / denom)
    return ret

def inner_sum_ref_full(lim, pax, pbx, px, l1, l2, eps):
    ret = [0.0] * lim
    for i in range(lim):
        for r in range(lim):
            for u in range(lim):
                num = (-1)**i * binomial_prefactor_ref(i, l1, l2, pax, pbx) * (-1)**u * factorial(i) * np.pow(px, i - 2*r - 2*u) * np.pow(eps, r+u)
                denom = factorial(r) * factorial(u) * factorial(i - 2*r - 2*u)
                ret[i - 2*r - u] += float(num / denom)
    return ret

inner_sum_ref = inner_sum_ref_restricted

### setup ###
a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 1, 2, 0.2, 0.3, 0.1
a2, l2, m2, n2, x2, y2, z2 =  0.1, 1, 1, 0, 10., 0.1, 0.5
nx, ny, nz = 10., 2, 3.

gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )
nuc = jnp.array([nx, ny, nz])

l_max = max(l1, m1, n1, l2, m2, n2) + 1
t_arr = jnp.arange(2*l_max)
l_arr = t_arr

# pack gaussians
ra, la, aa, rb, lb, ab, g, p, rap, rbp = pack_gaussian_pair(gaussian1, gaussian2)

# derived quantity
eps = 1/(4*g)

# relative position of center and nucleus    
cp = p - nuc
rcp2 = jnp.pow(jnp.linalg.norm(cp), 2)

### work ###
# N x 3 arr
d1 = p - gaussian1[:3]
d2 = p - gaussian2[:3]
bf = binomial_prefactor(l_arr, gaussian1.at[:3].set(d1), gaussian2.at[:3].set(d2), l_arr)
# bf = jnp.ones_like(bf)
a_arr = (factorial(l_arr) * jnp.pow(-1, l_arr))[:, None] * bf

# N arr
b_arr = jnp.pow(eps, l_arr) * 1 / factorial(l_arr)

# TODO: size used => no JIT!
b_arr = jnp.insert(arr = b_arr, obj = jnp.arange(0, b_arr.size) + 1, values = 0)

# N arr
c_arr = jnp.pow(-1 * eps, l_arr) / factorial(l_arr)

# M x N x 3 arr
d_arr = jax.vmap(jax.vmap(lambda I, u : jnp.pow(cp, I - u) / factorial(I - u), in_axes = (0, None)), in_axes=(None,0))(l_arr, l_arr)

# M x N x 3 arr
c_arr = d_arr * c_arr[:, None, None]
c_arr = jnp.nan_to_num(c_arr, posinf = 1, neginf = 1)

### compute with conventional loops ##

limx = limy = limz = 10
gx_val = nuclear_inner_loop(limx, 0, a_arr, b_arr, c_arr)
gy_val = nuclear_inner_loop(limy, 1, a_arr, b_arr, c_arr)
gz_val = nuclear_inner_loop(limz, 2, a_arr, b_arr, c_arr)

conv = jnp.convolve(jnp.convolve(gx_val, gy_val), gz_val)

f_arr = boys_fun( jnp.arange(conv.size), rcp2 * g )


# overall prefactor
rab2 = jnp.pow(jnp.linalg.norm(rb - ra), 2)
prefac = 2*jnp.pi/g * jnp.exp(-aa*ab*rab2/g)

# b * c
# first convolution N x N x 3, 2N => N x 3 x (3N - 1)
x = jax.vmap(lambda y : jax.vmap(lambda x : jnp.convolve(b_arr, x))(y.T))(c_arr)

# A = a . b * c
# second convolution N x 3 x (3N - 1), N x 3 => N x 3 x (4N - 2)
# vmapping over last axis of x and then Cartesian axis of x and a_arr
final_arrs = jax.vmap(lambda x1 : jax.vmap(jnp.convolve, in_axes=(0,0))(x1, a_arr.T) )(x)

# skip strictly positive indices N x 3 x (4N - 2) => 3 x N
final_arrs = jnp.diagonal(final_arrs[..., (x.shape[-1]-1):], axis1=0, axis2=2)

# final layer is convolution of cartesian arrays => N-dim
conv = jnp.convolve(final_arrs[0], jnp.convolve(final_arrs[1], final_arrs[2]))

# function array over unique range
f_arr = boys_fun( jnp.arange(conv.size), rcp2 * g )

# raw sum
# res = prefac * f_arr @ (conv+1)


### reference ###
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

ret = 0.0    
for I in range(limx):
    for J in range(limy):
        for K in range(limz):
            ret += boys_fun(I+J+K, dcp * g) * gx[I] * gy[J] * gz[K]

prefac = -2*np.pi / g * np.exp(-aa*ab*np.linalg.norm(ra-rb)**2/g)
ret *= prefac
