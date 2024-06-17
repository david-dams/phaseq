import time
import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma, factorial

jax.config.update("jax_enable_x64", True)

### ELEMENTARY FUNCTIONS ###
# TODO: make function, precomputed values for factorial
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
    """laplace operator between primitive gaussians
    
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
def nuclear_pack(a, b, c, d):
    return

def nuclear_convolve_first(a, b, c):
    return

def nuclear_convolve_second(a, b, c):
    return

def nuclear( l_arr, gaussian1, gaussian2, t_arr, nuc):
    """nuclear potential term (single nucleus) between primitive gaussians
    
    Args:
        l_arr : range of angular momenta, precomputed before simulation
        gaussian1, gaussian2 : array representations of primitive gaussians
        t_arr : dummy array for summation, must range from 2*min(l_arr) to 2*max(l_arr), precomputed before simulation
        nuc : array holding nuclear position

    Returns:
        float, overlap    
    """
    
    # unpack primitive gaussians
    ra, la, aa = gaussian1[:3], gaussian1[3:6], gaussian1[-1]
    rb, lb, ab = gaussian2[:3], gaussian2[3:6], gaussian2[-1]

    # "packing" layer: map a,b,c,d to a, b', c'    
    a, b_prime, c_prime = nuclear_pack(a,b,c,d)

    # first convolution layer
    a_arr = nuclear_convolve_first(a, b_prime, c_prime)

    # second convolution layer
    out = nuclear_convolve_second(a_arr, a_arr, a_arr)

    # dot product
    f_arr = ...
    return f @ out

def interaction(l_arr, gaussian1, gaussian2, gaussian3, gaussian4, t_arr):
    """interaction term between primitive gaussians

    Args:
        l_arr : range of angular momenta, precomputed before simulation
        gaussian1, gaussian2 : array representations of primitive gaussians
        t_arr : dummy array for summation, must range from 2*min(l_arr) to 2*max(l_arr), precomputed before simulation
        nuc : array holding nuclear position
    
    Returns:
       float
    """
    
    cs1, alphas1, lmn1, ps1, cs2, alphas2, lmn2, ps2, cs3, alphas3, lmn3, ps3, cs4, alphas4, lmn4, ps4 = orb1[:m], orb1[m:2*m], orb1[-6:-3], orb1[-3:], orb2[:m], orb2[m:2*m], orb2[-6:-3], orb2[-3:], orb3[:m], orb3[m:2*m], orb3[-6:-3], orb3[-3:], orb4[:m], orb4[m:2*m], orb4[-6:-3], orb4[-3:]

    
    # TODO: contraction of beta and gamma gives tensor of dimension N x N x N x N
    bg = repulsion_beta_times_gamma()

    # TODO: all possible orbital combinations
    orb_combinations = combinations
    
    # reduce memory hunger by turning vmap into map
    return jax.vmap(lambda orbs : tensor_element(orbs, angular_momenta, angular_momenta_index_list))(orb_combinations).sum()

def update_positions(orbitals, nuclei, orbitals_to_nuclei):
    """Ensures that orbital positions match their nuclei."""    
    return orbitals.at[:, :3].set(nuclei[orbitals_to_nuclei, :3])

# convention used for built-in gaussian basis: tuples have the form (coefficients, alphas, lmn), where lmn is exponent of cartesian coordinates x,y,z
sto_3g = {
    "pz" : (jnp.array([ 0.155916, 0.607684, 0.391957 ]), 
            jnp.array( [ 2.941249, 0.683483, 0.22229 ]),
            jnp.array( [ 0,0,1 ]) )
    }

if __name__ == '__main__':
    pass




