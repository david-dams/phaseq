import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma, factorial

jax.config.update("jax_enable_x64", True)

### ELEMENTARY FUNCTIONS ###
def boys_fun(index, arg):
    """computes the boys function for the specified index and scalar argument"""
    arg += (jnp.abs(arg) < 0.00000001) * 0.00000001
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

def kernel_to_matrix(kernel, imax, imax_lim):
    """Helper function for vectorized evaluation of certain "restricted cross-correlations" occuring in nuclear and interaction matrix elements.

    Args:
      kernel : N - array containing kernel elements of the restricted cross-correlation
      imax   : int, maximum combined angular momentum, such that N = imax // 2, chose as static arg of value 2*(maximum angular momentum of all gaussians) for JIT compilation
      imax_lim : int, used to determine the actual summation limit
    
    Returns:
       imax x imax array

    Allows vectorized evaluation of an expression like $T(I) =  \sum\limits_{i - 2r = I, r \leq i/2} a(i) b(r)$ where i/2 denotes integer division as 
    
    T = B @ a
    
    where B is a matrix. This occurs in evaluation nuclear and interaction matrix elements.

    This works as follows:

    1. Inflate the array b' of b-values such that b'[2r] = b[r] and b[2r+1] = 0.
    2. Then, since $\sum\limits_{i - j = I, j \leq i} a[i] b'[j] = \sum\limits_{i, I \geq 0} a[i] b'[i - I]$ define a matrix B such that B[I, i] = b'[i - I].
    """
    # zero-out elements
    kernel *= (jnp.arange(kernel.size) <= imax_lim // 2)
    
    # "inflate" g such that g[2*r] = g(r)
    kernel = jnp.insert(arr = kernel, obj = jnp.arange(kernel.size) + 1, values = 0)
                
    # Create a 2D array of indices
    row_indices = jnp.arange(imax)[:, None]  # Shape: (N, 1)
    col_indices = jnp.arange(imax)[None, :]  # Shape: (1, n)
    
    # Compute the shifted indices
    indices = col_indices - row_indices

    # reshape g such that gs[L, i] = gs[i - L]
    # Gather the elements using the shifted indices, masking out "wrap-around" indices
    M = kernel[indices] * (indices >= 0) * (col_indices > (imax_lim-imax))
    return M

def norm(gaussian):
    nom = jnp.pow(2.0, 2.0 * gaussian[3:6].sum() + 1.5) * jnp.pow(gaussian[-1], gaussian[3:6].sum() + 1.5)
    # (2n - 1)!! = 2**n * G(n + 0.5) / sqrt(pi)
    denom = jnp.prod(jnp.pow(2.0, gaussian[3:6])*gamma(gaussian[3:6] + 0.5) / jnp.sqrt(jnp.pi)) * jnp.pow(jnp.pi, 1.5)
    return jnp.sqrt(nom / denom)

### OVERLAP ###
def overlap(gaussian1, gaussian2, l_arr, t_arr):
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

    # double factorial array divided by 2**i    
    c_arr = gamma(l_arr+0.5) / (jnp.sqrt(jnp.pi) * jnp.pow(g, l_arr))

    return a * (b_arr * c_arr[:, None]).sum(axis=0).prod()


### KINETIC ###
def kinetic(gaussian1, gaussian2, l_arr, t_arr):
    """laplace operator between primitive gaussians. decomposes into overlaps and is just dumbly implemented as such.    
    
    Args:
        gaussian1, gaussian2 : array representations of primitive gaussians
        l_arr : range of angular momenta, precomputed before simulation
        t_arr : dummy array for summation, must range from 2*min(l_arr-2) to 2*max(l_arr+2), precomputed before simulation

    Returns:
        float, kinetic matrix elements
    """    
    element = gaussian2[-1] * (2.0 * gaussian2[3:6].sum() + 3.0) * overlap(gaussian1, gaussian2, l_arr, t_arr)
    
    element += -2.0 * jnp.pow(gaussian2[-1], 2) * (overlap(gaussian1, gaussian2.at[3].set(gaussian2[3]+2), l_arr, t_arr) +
                                                overlap(gaussian1, gaussian2.at[4].set(gaussian2[4]+2), l_arr, t_arr) +
                                                overlap(gaussian1, gaussian2.at[5].set(gaussian2[5]+2), l_arr, t_arr) )

    fac = gaussian2[3:6] * (gaussian2[3:6] - 1)
    element += -0.5 * ( fac[0] * overlap(gaussian1, gaussian2.at[3].set(gaussian2[3]-2), l_arr, t_arr) +
                     fac[1] * overlap(gaussian1, gaussian2.at[4].set(gaussian2[4]-2), l_arr, t_arr) +
                     fac[2] * overlap(gaussian1, gaussian2.at[5].set(gaussian2[5]-2), l_arr, t_arr) )

    return element


### NUCLEAR ###
def nuclear(gaussian1, gaussian2, nuc, l_max=0):
    """nuclear potential term (single nucleus) between primitive gaussians.
    
    Args:    
        gaussian1, gaussian2 : array representations of primitive gaussians
        nuc : 3 array, nucleus position
        l_max : int, maximum combined angular momentum. Pick maximum possible angular momentum of all primitives and pass as static when JIT-ing this function.

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
    # l_max = jnp.max(gaussian1[3:6] + gaussian2[3:6]) + 1
    l_range = jnp.arange(l_max) # size is l1 + l2 + 1
    pa, pb = p - gaussian1[:3], p - gaussian2[:3]
    bf1 = binomial_prefactor(l_range, gaussian1.at[:3].set(pa), gaussian2.at[:3].set(pb), l_range)
    iterm = (factorial(l_range) * jnp.pow(-1, l_range))[:, None] * bf1
    l_range_half = jnp.arange(l_max // 2)
    rterm = jnp.pow(eps, l_range_half) / factorial(l_range_half)

    # components of big matrix
    L_range, I_range = l_range, l_range
    t1 = jnp.pow(-eps, L_range - I_range[:, None])
    t2 = factorial(2*I_range[:, None] - L_range) * factorial(-I_range[:, None] + L_range)
    t3 = jnp.pow(cp[:, None, None], 2*I_range[:, None] - L_range)
    c = t1 * (I_range[:, None] >= L_range//2) * jnp.nan_to_num(1/t2, nan= 0, posinf = 0, neginf = 0) * jnp.nan_to_num(t3, nan= 0, posinf = 0, neginf = 0) 

    # convolution
    As = []

    # TODO: uff
    for i in range(3):
        lim = gaussian1[3:6][i] + gaussian2[3:6][i] + 1
        mat = kernel_to_matrix(rterm, l_max, lim)
        v = iterm[:, i]
        a = mat @ v
        res = c[i] @ a                
        As.append(res)

    conv = jnp.convolve(jnp.convolve(As[0], As[1]), As[2])

    # each array has size l_max => total size is double conv => l_max + l_max - 1 + l_max - 1
    conv_size = 3*l_max - 2
    
    # function array over unique range
    f_arr = boys_fun( jnp.arange(conv_size), rcp2 * g )

    # raw sum
    res = f_arr @ conv
    
    rab2 = jnp.pow(jnp.linalg.norm(rb - ra), 2)
    prefac = 2*jnp.pi/g * jnp.exp(-aa*ab*rab2/g)
    
    return -prefac * res

### INTERACTION ###    
def interaction_d_matrix(K_range, I_range, p, delta):
    """computes the matrix: $e_{I, K} &= \frac{ K! (-)^{K-I} p_x^{2I - K}}{(K-I)!(2I -K)!\delta^{I}}$"""

    t1 = factorial(K_range) / jnp.pow(delta, I_range)[:, None]
    t2 =  factorial(2*I_range[:, None] - K_range) * factorial(K_range - I_range[:, None]) 
    t3 = jnp.pow(-1, K_range - I_range[:, None]) * jnp.pow(p[:, None, None], 2*I_range[:, None] - K_range)
    
    return t1 * jnp.nan_to_num(t3/t2, posinf = 0, neginf = 0)
    
def interaction(gaussian1, gaussian2, gaussian3, gaussian4):    
    # unpack gaussians
    a1, a2 = gaussian1[-1], gaussian2[-1]
    a3, a4 = gaussian3[-1], gaussian4[-1]

    # gaussian centers
    gamma1 = a1 + a2
    p = (a1 * gaussian1[:3] + a2 * gaussian2[:3]) / gamma1
    gamma2 = a3 + a4
    q = (a3 * gaussian3[:3] + a4 * gaussian4[:3]) / gamma2
    delta = 1/(4*gamma1) + 1/(4*gamma2)
    
    # distance center-center
    center_center = q - p

    # for each direction (x,y,z) we have one loop over indices.
    # this loop runs as follows
    # 0 <= i1 <= l1 + l2
    # 0 <= i2 <= l3 + l4
    # 0 <= r1 <= i1 // 2
    # 0 <= r2 <= i2 // 2
    # 0 <= u <= (i1+i2)//2 - 2(r1+r2)
    # where li is i-th the angular momentum in x (or y,z).
    # The loop can be written as a function T as follows
    # T(I) = sum_{i1 + i2 - 2(r1 + r2) - u = I} a(i1, r1) b(i2, r2) c(i1 + i2 - 2(r1 + r2), u)
    # we can replace the a, b functions with a vector B and reshape the c function into a square matrix to arrive at
    # T(I) = \sum_K A_{I, K} B_K => T = A @ B
    # where T is a vector of size l1 + l2 + l3 + l4 + 1
        
    # precompute arrays for the big vector B
    # this vector is computed as the convolution of two arrays a, b of shape l1 + l2 + 1, l3 + l4 + 1
    # we precompute quantities over the largest index range, given by max(l1+l2, m1+m2, n1+n2) + 1

    # precompute for a array
    l_max1 = jnp.max(gaussian1[3:6] + gaussian2[3:6])
    l_range = jnp.arange(l_max1 + 1) # size is l1 + l2 + 1
    pa, pb = p - gaussian1[:3], p - gaussian2[:3]
    bf1 = binomial_prefactor(l_range, gaussian1.at[:3].set(pa), gaussian2.at[:3].set(pb), l_range)
    facgamma1 = factorial(l_range) / jnp.pow(4*gamma1, l_range)
    rterm1 = 1/facgamma1
    iterm1 = facgamma1[:, None] * bf1
    fac1 = 1 / factorial(l_range)

    # precompute for b array
    l_max2 = jnp.max(gaussian3[3:6] + gaussian4[3:6])
    l_range = jnp.arange(l_max2 + 1)
    qc, qd = q - gaussian3[:3], q - gaussian4[:3]
    bf2 = binomial_prefactor(l_range, gaussian3.at[:3].set(qc), gaussian4.at[:3].set(qd), l_range)
    facgamma2 = factorial(l_range) / jnp.pow(4*gamma2, l_range)    
    rterm2 = 1/facgamma2
    iterm2 = (facgamma2 * jnp.pow(-1, l_range))[:, None] * bf2
    fac2 = 1 / factorial(l_range)
    
    # precompute the matrix A
    K_range = jnp.arange(l_max1 + l_max2 + 1) # size is l1 + l2 + l3 + l4 + 1
    I_range = jnp.arange(l_max1 + l_max2 + 1)
    d = interaction_d_matrix(K_range, I_range, center_center, delta)

    # convolution
    As = []

    # TODO: uff
    for i in range(3):
        lim = int(gaussian1[3:6][i] + gaussian2[3:6][i]) + 1
        v = iterm1[:lim, i]
        a = fac1[:v.size] * (kernel_to_matrix(rterm1, v.size) @ v)

        lim = int(gaussian3[3:6][i] + gaussian4[3:6][i]) + 1
        v = iterm2[:lim, i]
        b = fac2[:v.size] * (kernel_to_matrix(rterm2, v.size) @ v)

        c = jnp.convolve(a, b)
        res = d[i, :c.size, :c.size] @ c
        As.append(res)
        
    conv = jnp.convolve(As[0], jnp.convolve(As[1], As[2]))
    
    # Boys-function
    arg = jnp.linalg.norm(p-q)**2 / (4 * delta)
    f_arr = boys_fun( jnp.arange(conv.size), arg )

    # global prefactor
    rab2 = jnp.linalg.norm(gaussian1[:3]-gaussian2[:3])**2
    rcd2 = jnp.linalg.norm(gaussian3[:3]-gaussian4[:3])**2
    prefac = 2.0 * jnp.pow(jnp.pi, 2.5) / (gamma1 * gamma2 * jnp.sqrt(gamma1 + gamma2)) * jnp.exp(-a1*a2*rab2/gamma1) * jnp.exp(-a3*a4*rcd2/gamma2)

    return prefac * (f_arr @ conv)
