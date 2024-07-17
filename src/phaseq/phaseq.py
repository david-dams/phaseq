import time
import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma, factorial

jax.config.update("jax_enable_x64", True)

### ELEMENTARY FUNCTIONS ###
def boys_fun(index, arg):
    """computes the boys function for the specified index and scalar argument"""
    return gammainc(index+0.5, arg) * gamma(index+0.5) * 0.5 * jnp.pow(arg,-index-0.5) 

def binomial(n, m):
    return jnp.nan_to_num(factorial(n) / (factorial(n - m) * factorial(m)), posinf=1, neginf=1)

def binomial_prefactor(s_arr, gaussian1, gaussian2, t_arr):
    """binomial prefactor array. each element is computed for an "external" angular momentum by summing over a range of "internal"  angular momenta.

    Args:
        s_arr : N-dim array; external angular momentum range
        gaussian1, gaussian2 : array representation of gaussians containing 3-dim position vectors
        t_arr : M-dim array; internal angular momentum range to be summed over
       
    Returns:
        array of shape N x 3
    """

    # s x t dim array    
    st = s_arr[:, None] - t_arr

    # s x 3 x t dim array
    lower_limit = t_arr[None, None, :] <= (jnp.minimum(s_arr[:, None], gaussian2[None, 3:6]))[:, :, None]
    upper_limit = (s_arr[:, None] - gaussian1[None, 3:6])[:, :, None] <= t_arr[None, None, :]
    mask = lower_limit * upper_limit
    st_terms = binomial(gaussian1[None, 3:6, None], st[:, None, :]) * jnp.pow(gaussian1[None, :3, None], gaussian1[None, 3:6, None] - st[:, None, :])

    # 3 x t array
    t_terms = binomial(gaussian2[3:6, None], t_arr) * jnp.pow(gaussian2[:3, None], gaussian2[3:6,None] - t_arr)
    
    # contract to s x 3 array
    contracted = jnp.einsum('it, sit->si', jnp.nan_to_num(t_terms, posinf=0, neginf=0), jnp.nan_to_num(st_terms, posinf=0, neginf=0) * mask)

    return contracted

def pack_gaussian_pair(gaussian1, gaussian2):
    # unpack primitive gaussians
    ra, la, aa = gaussian1[:3], gaussian1[3:6], gaussian1[-1]
    rb, lb, ab = gaussian2[:3], gaussian2[3:6], gaussian2[-1]

    # add gaussians
    g = aa + ab
    p = (aa*ra + ab*rb) / g    
    rap = jnp.linalg.norm(ra-p)**2
    rbp = jnp.linalg.norm(rb-p)**2

    return ra, la, aa, rb, lb, ab, g, p, rap, rbp

### OVERLAP ###
def overlap(l_arr, gaussian1, gaussian2, t_arr):
    """overlap between primitive gaussians

    in abstract terms, the overlap graph is a "packing layer" transforming primitive gaussians into arrays fed into a "contraction layer"
    
    Args:
        l_arr : range of angular momenta, precomputed before simulation
        gaussian1, gaussian2 : array representations of primitive gaussians
        t_arr : dummy array for summation, must range from 2*min(l_arr) to 2*max(l_arr), precomputed before simulation

    Returns:
        float, overlap    
    """

    # pack gaussians
    ra, la, aa, rb, lb, ab, g, p, rap, rbp = pack_gaussian_pair(gaussian1, gaussian2)
    
    # prefactor
    a = jnp.pow(jnp.pi / g, 1.5)  * jnp.exp( -aa*ab * jnp.linalg.norm(ra-rb)**2 / g)    
    
    # summation limits
    l_limits = jnp.floor((la + lb) / 2)

    # l x 3 array
    d1 = p - gaussian1[:3]
    d2 = p - gaussian2[:3]

    # in the loopy formulation, we sum over 2*i => array, where arr[i] ~ f(2*i)
    b_arr = binomial_prefactor(2*l_arr, gaussian1.at[:3].set(d1), gaussian2.at[:3].set(d2), t_arr) * (l_arr[:, None] <= l_limits)

    # double factorial array
    c_arr = gamma(l_arr+0.5) / (jnp.sqrt(jnp.pi) * jnp.pow(g, l_arr))

    return a * (b_arr * c_arr[:, None]).sum(axis=0).prod()
    
### KINETIC ###
def kinetic(l_arr, gaussian1, gaussian2, t_arr):
    """laplace operator between primitive gaussians. decomposes into overlaps and is just dumbly implemented as such.    
    
    Args:
        l_arr : range of angular momenta, precomputed before simulation
        gaussian1, gaussian2 : array representations of primitive gaussians
        t_arr : dummy array for summation, must range from 2*min(l_arr-2) to 2*max(l_arr+2), precomputed before simulation

    Returns:
        float, kinetic matrix elements
    """    
    element = gaussian2[-1] * (2.0 * gaussian2[3:6].sum() + 3.0) * overlap(l_arr, gaussian1, gaussian2, t_arr)
    
    element += -2.0 * jnp.pow(gaussian2[-1], 2) * (overlap(l_arr, gaussian1, gaussian2.at[3].set(gaussian2[3]+2), t_arr) +
                                                overlap(l_arr, gaussian1, gaussian2.at[4].set(gaussian2[4]+2), t_arr) +
                                                overlap(l_arr, gaussian1, gaussian2.at[5].set(gaussian2[5]+2), t_arr) )

    fac = gaussian2[3:6] * (gaussian2[3:6] - 1)
    element += -0.5 * ( fac[0] * overlap(l_arr, gaussian1, gaussian2.at[3].set(gaussian2[3]-2), t_arr) +
                     fac[1] * overlap(l_arr, gaussian1, gaussian2.at[4].set(gaussian2[4]-2), t_arr) +
                     fac[2] * overlap(l_arr, gaussian1, gaussian2.at[5].set(gaussian2[5]-2), t_arr) )

    return element


### NUCLEAR ###
def nuclear_pack(gaussian1, gaussian2, nuc, l_arr):
    """Packs gaussians and nuclear position into arrays    

    Args:
        gaussian1, gaussian2 : array representations of primitive gaussians
        l_arr : range of combined angular momenta from 0 to 2*l_max
        nuc : 3 array, nucleus position

    Returns:
        a  : N x 3 array 
        b' : N array 
        c' : N x N x 3 array 
        g : float
    """

    # pack gaussians
    ra, la, aa, rb, lb, ab, g, p, rap, rbp = pack_gaussian_pair(gaussian1, gaussian2)

    # derived quantity
    eps = 1/(4*g)

    # relative position of center and nucleus    
    cp = p - nuc
    rcp2 = jnp.pow(jnp.linalg.norm(cp), 2)

    # N x 3 arr
    d1 = p - gaussian1[:3]
    d2 = p - gaussian2[:3]
    a_arr = (factorial(l_arr) * jnp.pow(-1, l_arr))[:, None] * binomial_prefactor(l_arr, gaussian1.at[:3].set(d1), gaussian2.at[:3].set(d2), l_arr)

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

    # overall prefactor
    rab2 = jnp.pow(jnp.linalg.norm(rb - ra), 2)
    prefac = 2*jnp.pi/g * jnp.exp(-aa*ab*rab2/g)
    
    return a_arr, b_arr, jnp.nan_to_num(c_arr, posinf = 1, neginf = 1), g, prefac, rcp2

# TODO: transform to conv with kernel
def nuclear_inner_loop(lim, dimension, a_arr, b_arr, c_arr):
    ret = [0.0] * lim
    for i in range(lim):
        for r in range(i//2+1):
            for u in range((i-2*r)//2+1):        
                I = i - 2*r - u
                ret[I] += float(a_arr[i, dimension] * b_arr[2*r] * c_arr[u, I, dimension])
    return jnp.array(ret)

def nuclear(gaussian1, gaussian2, nuc, l_arr):
    """nuclear potential term (single nucleus) between primitive gaussians.
    
    Args:    
        gaussian1, gaussian2 : array representations of primitive gaussians
        l_arr : range of angular momenta from 0 to 2*l_max
        nuc : 3 array, nucleus position

    Returns:
        float, overlap    
    """
    # the innermost loop is $r_{I \geq 0} = \sum_{i - j - k} a_i b'_j c'_k$
    # this can be expressed as $r_{I \geq 0} = Conv[a, Conv[b', c']]_{I \geq 0}$
    
    # "packing" layer    
    a_arr, b_arr, c_arr, g, prefac, rcp2 = nuclear_pack(gaussian1, gaussian2, nuc, l_arr)

    # TODO: urgh
    lmn = gaussian1[3:6] + gaussian2[3:6] + 1
    gx_val = nuclear_inner_loop(int(lmn[0]), 0, a_arr, b_arr, c_arr)
    gy_val = nuclear_inner_loop(int(lmn[1]), 1, a_arr, b_arr, c_arr)
    gz_val = nuclear_inner_loop(int(lmn[2]), 2, a_arr, b_arr, c_arr)

    conv = jnp.convolve(jnp.convolve(gx_val, gy_val), gz_val)

    # function array over unique range
    f_arr = boys_fun( jnp.arange(conv.size), rcp2 * g )

    # raw sum
    res = f_arr @ conv

    return -prefac * res

### INTERACTION ###
def interaction_pack(gaussian1, gaussian2, gaussian3, gaussian4, nuc, l_arr, u_arr, u_t_arr):
    """Packs gaussians and nuclear position into arrays    

    Args:
        gaussian1, gaussian2, gaussian3, gaussian4 : array representations of primitive gaussians
        l_arr : range of angular momenta, precomputed before simulation
        t_arr : dummy array for summation, must range from 2*min(l_arr) to 2*max(l_arr), precomputed before simulation
        nuc : array holding nuclear position

    Returns:
        alpha : N array containing the inverse factorial
        beta' : N x 3 array containing the binomial prefactor scaled with the factorial
        gamma : N - array containing a combinatorial factor
        c : N x m x 3 array containing multinomial geometric factor
    """
    
    eps = 1/(4*g)
    
    # TODO: what args?
    a = factorial(u_arr) * jnp.pow(-1, l_range) * binomial_prefactor(gaussian1, gaussian2, u_arr, u_t_arr)
    
    b = jnp.pow(eps, u_arr) * 1 / factorial(u_arr)
    b = jnp.insert(arr = b, obj = jnp.arange(0, b.size, 2), values = 0)[::-1]

    c = jnp.pow(-1 * eps, u_arr) / factorial(u_arr)

    d = jax.vmap(lambda I, u : jnp.pow(p, I - u) / factorial(I - u), l_arr, u_arr)

    c = d * c[None,:]
    
    return a, b, c[:, ::-1]

def interaction(gaussian1, gaussian2, gaussian3, gaussian4, l_arr, t_arr):
    """interaction term between primitive gaussians

    Args:
        l_arr : range of angular momenta, precomputed before simulation
        gaussian1, gaussian2 : array representations of primitive gaussians
        t_arr : dummy array for summation, must range from 2*min(l_arr) to 2*max(l_arr), precomputed before simulation
        nuc : array holding nuclear position
    
    Returns:
       float
    """

    # first layer: unpack
    alpha, beta, gamma, c = pack_interaction()

    # second layer: convolution and hadamard product
    a_arr = gamma * jax.vmap(nuclear_convolve_first)(alpha, beta)

    # third layer: convolution and dot product
    out = nuclear_convolve_second(a_arr[:,0], a_arr[:,1], a_arr[:,2])
    f_arr = boys_fun(l_range)
    return f @ out

def nuclear_matrix(arr1, arr2, nuc):

    gaussian1 = jnp.concatenate( [arr1, jnp.array( [1, 0, 0, 0.1] )] )
    gaussian2 = jnp.concatenate( [arr2, jnp.array( [1, 0, 0, 0.1] )] )

    # array of combined angular momenta
    l_max = 1 #max(gaussian1[3:6].max(), gaussian2[3:6].max()) 
    l_arr = jnp.arange(2*l_max+1)
    
    return nuclear(gaussian1, gaussian2, nuc, l_arr)
