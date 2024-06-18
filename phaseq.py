import time
import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma, factorial

jax.config.update("jax_enable_x64", True)

# TODO: unpacking primitive gaussians should happen in separate function
# TODO: make function, precomputed values for factorial

### ELEMENTARY FUNCTIONS ###
DOUBLE_FACTORIAL = jnp.array([1,1,2,3,8,15,48,105,384,945,3840,10395,46080,135135,645120,2027025])
def double_factorial(n):
    return
    
def gamma_fun(s, x):
    return gammainc(s+0.5, x) * gamma(s+0.5) * 0.5 * jnp.pow(x,-s-0.5) 
    
def binomial(n, m):
    return (n > 0) * (m > 0) * (n > m) * (factorial(n) / (factorial(n - m) * factorial(m)) - 1) + 1

def binomial_prefactor(s_arr, gaussian1, gaussian2, t_arr):
    """binomial prefactor array over a combined angular momentum range for two gaussians.

    Args:
        s_arr : 
        gaussian1, gaussian2 : 
        t_arr :
       
    Returns:
         array of shape N x 3, where N is the size of s_arr and 3 is the Cartesian dimension
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

    # unpack primitive gaussians
    ra, la, aa = gaussian1[:3], gaussian1[3:6], gaussian1[-1]
    rb, lb, ab = gaussian2[:3], gaussian2[3:6], gaussian2[-1]

    # add gaussians
    g = aa + ab
    p = (aa*ra + ab*rb) / g    
    rap = jnp.linalg.norm(ra-p)**2
    rbp = jnp.linalg.norm(rb-p)**2
    
    # prefactor
    a = jnp.pow(jnp.pi / g, 1.5)  * jnp.exp( -aa*ab * jnp.linalg.norm(ra-rb)**2 / g)    
    
    # summation limits
    l_limits = jnp.floor((la + lb) / 2)

    # l x 3 array
    b_arr = binomial_prefactor(2*l_arr, gaussian1, gaussian2, t_arr) * (l_arr[:, None] >= l_limits)

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

    fac = gaussian2[3:6] * (gaussian[3:6] - 1)
    element += -0.5 * ( fac[0] * overlap(l_arr, gaussian1, gaussian2.at[3].set(gaussian2[3]-2), t_arr) +
                     fac[1] * overlap(l_arr, gaussian1, gaussian2.at[4].set(gaussian2[4]-2), t_arr) +
                     fac[2] * overlap(l_arr, gaussian1, gaussian2.at[5].set(gaussian2[5]-2), t_arr) )

    return element


### NUCLEAR ###
# nuclear loop is basically
# L = \sum_{n} F_n \sum_{I+J+K = n} G_I G_J G_K
# I \in [0, l_1 + l_2]
# J \in [0, m_1 + m_2]
# K \in [0, n_1 + n_2]
# padded: n, I, J, K \in [0, 6*l_max]
# this can be expressed as
# L = F @ Conv_3[A(l_1+l_2, r_x), A(m_1+m_2, r_y), A(n_1+n_2, r_z)]
# A(l_1+l_2, r_x) = Conv_3[a, b, c[:]]
# such that len(A) = 6 * l_max
# len(a) = len(b) = 2*l_max
# c.size = (6*l_max, l_max)
# A needs to be masked as follows: [0, l_1+l_2]

def nuclear_pack(gaussian1, gaussian2, nuc, l_arr, u_arr):
    """Packs gaussians and nuclear position into arrays    

    Args:
        gaussian1, gaussian2 : array representations of primitive gaussians
        l_arr : range of combined angular momenta from 0 to 6*l_max
        u_arr : range of isolated angular momenta from 0 to 2*l_max
        nuc : 3 array, nucleus position

    Returns:
        a : m x 3 array containing the binomial prefactor for all isolated angular momenta and Cartesian directions
        b' : m array containing a factor scaling with the inverse factorial
        c' : N x m x 3 array containing combined and isolated angular momentum arrays
    """

    # unpack primitive gaussians
    ra, la, aa = gaussian1[:3], gaussian1[3:6], gaussian1[-1]
    rb, lb, ab = gaussian2[:3], gaussian2[3:6], gaussian2[-1]

    # add gaussians
    g = aa + ab
    p = (aa*ra + ab*rb) / g    
    rap = jnp.linalg.norm(ra-p)**2
    rbp = jnp.linalg.norm(rb-p)**2
    eps = 1/(4*g)

    # relative position of center and nucleus    
    pc = p - nuc

    # N x 3 arr
    a = (factorial(u_arr) * jnp.pow(-1, u_arr))[:, None] * binomial_prefactor(u_arr, gaussian1, gaussian2, u_arr)

    # N arr
    b = jnp.pow(eps, u_arr) * 1 / factorial(u_arr)
    b = jnp.insert(arr = b, obj = jnp.arange(0, b.size, 2), values = 0)[::-1]

    # N arr
    c = jnp.pow(-1 * eps, u_arr) / factorial(u_arr)
    
    # M x N x 3 arr
    d = jax.vmap(jax.vmap(lambda I, u : jnp.pow(pc, I - u) / factorial(I - u), in_axes = (0, None)), in_axes=(None,0))(l_arr, u_arr)
    
    # M x N x 3 arr
    c = d * c[:, None, None]    
    return a, b, jnp.nan_to_num(c[:, ::-1], posinf = 0, neginf = 0)


def nuclear(gaussian1, gaussian2, nuc, l_arr, u_arr):
    """nuclear potential term (single nucleus) between primitive gaussians.

    in abstract terms, it consists of a "packing" layer transforming primitive gaussians into arrays fed into a convolution layer which are again fed into a convolution layer and a dot product like so

    nuclear = conv(conv(pack(gaussians))) @ gamma_fun
    
    Args:    
        gaussian1, gaussian2 : array representations of primitive gaussians
        l_arr : range of combined angular momenta from 0 to 6*l_max
        u_arr : range of isolated angular momenta from 0 to 2*l_max
        nuc : 3 array, nucleus position

    Returns:
        float, overlap    
    """
    
    # unpack primitive gaussians
    ra, la, aa = gaussian1[:3], gaussian1[3:6], gaussian1[-1]
    rb, lb, ab = gaussian2[:3], gaussian2[3:6], gaussian2[-1]    

    # "packing" layer: map gaussians to a, b', c'    
    a, b_prime, c_prime = nuclear_pack(gaussian1, gaussian2, nuc, l_arr, u_arr)

    # first convolution layer along cartesian axis produces a N x 3 array
    a_arr = jax.vmap(nuclear_convolve_first)(a, b_prime, c_prime)

    # second convolution layer produces N-dim array
    out = nuclear_convolve_second(a_arr[:,0], a_arr[:,1], a_arr[:,2])

    # dot product
    f_arr = gamma_fun(l_range)
    return f @ out

# primitive gaussian is always [pos, lmn, alpha]
g1 = jnp.array( [-0.1, 0.3, 0.7, 2, 1, 3, 0.2] )
g2 = jnp.array( [0.1, 0.4, 0.1, 2, 0, 5, 0.1] )

nuc = jnp.arange(3) + 0.0

# array of combined angular momenta
l_max = (g1[3:6].max() + g2[3:6].max()) + 1
u_arr = jnp.arange(2*l_max)
l_arr = jnp.arange(6*l_max)
a,b,c = nuclear(g1, g2, nuc, l_arr, u_arr)

### INTERACTION ###
# interaction loop is basically
# L = \sum_{n} F_n \sum_{I+J+K = n} G_I G_J G_K
# I \in [0, l_1 + l_2 + l_3 + l_4]
# J \in [0, m_1 + m_2 + m_3 + m_4]
# K \in [0, n_1 + n_2 + n_3 + n_4]
# padded: n, I, J, K \in [0, 12*l_max]
# this can be expressed as
# L = F @ Conv_3[A(l_1+l_2+l_3+l_4, r_x), A(m_1+m_2+m_3+m_4, r_y), A(n_1+n_2+n_3+n_4, r_z)]
# A(l_1+l_2, r_x) = Conv_3[a, b, c[:]]
# b[i] = (-1)**i * a[i]
# a = gamma * Conv[alpha, beta]
# alpha_i = 1/i!
# beta_i \sim i! f_i
# c[i, j]

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
    f_arr = gamma_fun(l_range)
    return f @ out

# convention used for built-in gaussian basis: tuples have the form (coefficients, alphas, lmn), where lmn is exponent of cartesian coordinates x,y,z
sto_3g = {
    "pz" : (jnp.array([ 0.155916, 0.607684, 0.391957 ]), 
            jnp.array( [ 2.941249, 0.683483, 0.22229 ]),
            jnp.array( [ 0,0,1 ]) )
    }

if __name__ == '__main__':
    pass




