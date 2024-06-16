import time
import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gamma, factorial

### ELEMENTARY FUNCTIONS ###
def gamma_fun(s, x):
    return gammainc(s+0.5, x) * gamma(s+0.5) * 0.5 * jnp.pow(x,-s-0.5) 
    
def binomial(n, m):
    return (n > 0) * (m > 0) * (n > m) * (factorial(n) / (factorial(n - m) * factorial(m)) - 1) + 1
    
def binomial_prefactor(s, ia, ib, xpa, xpb):
    # get binomial arrays
    ia_terms = binomial(ia, s - ts) * jnp.pow(xpa, ia - s + ts)
    ib_terms = binomial(ib, ts) * jnp.pow(xpb, ib - ts)

    return ((ts < s+1) * (s-ia <= ts) * (ts <= ib) * ia_terms * ib_terms).sum()

def double_factorial(n):
    vals = jnp.array([1,1,2,3,8,15,48,105,384,945,3840,10395,46080,135135,645120,2027025])
    n_max = vals.size
    rng = jnp.arange(n_max)
    return (n > 0) * jnp.where(rng == n, vals[rng], 0).sum() + (n < 0)

def gaussian_norm(lmn, alphas):
    nom = jax.vmap(lambda a : jnp.pow(2.0, 2.0*(lmn.sum()) + 1.5) * jnp.pow(a, lmn.sum() + 1.5))(alphas)
    denom = (jax.vmap(lambda i : double_factorial(2*i-1))(lmn)).prod() * jnp.pow(jnp.pi, 1.5)
    return jnp.sqrt(nom / denom)

### OVERLAP ###
def get_overlap_1d(l_max_range):
    def body(val, i, l1, l2, x1, x2, gamma):
        return jax.lax.cond(i <  1 + jnp.floor(0.5*(l1+l2)),                            
                            lambda : val + binomial_prefactor(2*i, l1, l2, x1, x2) * double_factorial(2*i-1) / jnp.pow(2*gamma, i),
                            lambda : val)

    def wrapper(*params):
        return jax.lax.scan(lambda val, i : (body(val, i, *params), None), 0.0, l_max_range)[0]

    binomial_prefactor = get_binomial_prefactor(l_max_range)
    
    return wrapper

def get_overlap(l_max):
    def overlap(alpha1, lmn1, pos1, alpha2, lmn2, pos2):
        rab2 = jnp.linalg.norm(pos1-pos2)**2
        gamma = alpha1 + alpha2
        p = (alpha1*pos1 + alpha2*pos2) / gamma
        pre = jnp.pow(jnp.pi/gamma, 1.5) * jnp.exp(-alpha1*alpha2*rab2/gamma)

        vpa =  p - pos1
        vpb =  p - pos2

        wx = overlap_1d(lmn1[0], lmn2[0], vpa[0], vpb[0], gamma)
        wy = overlap_1d(lmn1[1], lmn2[1], vpa[1], vpb[1], gamma)
        wz = overlap_1d(lmn1[2], lmn2[2], vpa[2], vpb[2], gamma)
        
        return pre*wx*wy*wz

    l_max_range = jnp.arange(l_max)
    overlap_1d  = get_overlap_1d(l_max_range)
    
    return overlap


### KINETIC ###
def get_kinetic(l_max):
    
    def kinetic(alpha1, lmn1, pos1, alpha2, lmn2, pos2):
        term = alpha2 * (2.0 * lmn2.sum() + 3.0) * overlap(alpha1, lmn1, pos1, alpha2, lmn2, pos2)

        lmn2_inc = lmn2 + 2
        term += -2.0 * jnp.pow(alpha2, 2) * (overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[0].set(lmn2_inc[0]), pos2) +
                                               overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[1].set(lmn2_inc[1]), pos2) +
                                               overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[2].set(lmn2_inc[2]), pos2) )

        lmn2_dec = lmn2 - 2
        term += -0.5 * ( (lmn2[0] * (lmn2[0] - 1)) * overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[0].set(lmn2_dec[0]), pos2) +
                 lmn2[1]*(lmn2[1]-1) * overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[1].set(lmn2_dec[1]), pos2) +
                 lmn2[2]*(lmn2[2]-1) * overlap(alpha1, lmn1, pos1, alpha2, lmn2.at[2].set(lmn2_dec[2]), pos2) )
        
        return term

    l_max = l_max + 2
    overlap = get_overlap(l_max)
    
    return kinetic


### NUCLEAR ###
def get_a_array(l_max):

    def a_term(i, r, u, l1, l2, pa, pb, cp, g):
        return ( jnp.pow(-1,i) * binomial_prefactor(i,l1,l2,pa,pb)*
                 jnp.pow(-1,u) * factorial(i)*jnp.pow(cp,i-2*r-2*u)*
                 jnp.pow(0.25/g,r+u)/factorial(r)/factorial(u)/factorial(i-2*r-2*u) )
    
    def a_legal(iI, i, r, u, l1, l2):
        return jnp.logical_and( jnp.logical_and( jnp.logical_and(i < l1 + l2 + 1, r <= i//2),  u <= (i-2*r)//2),  iI == i - 2 * r - u )

    def a_wrapped(iI, i, r, u, l1, l2, pa, pb, cp, g):
        return jax.lax.cond(a_legal(iI, i, r, u, l1, l2), lambda: a_term(i, r, u, l1, l2, pa, pb, cp, g), lambda : 0.0) 

    def a_loop(iI, l1, l2, pa, pb, cp, g):
        return jax.lax.fori_loop(0, imax,
                lambda i, vx: vx + jax.lax.fori_loop(0, rmax,
                    lambda r, vy: vy + jax.lax.fori_loop(0, umax,
                        lambda u, vz: vz + a_wrapped(iI, i, r, u, l1, l2, pa, pb, cp, g),
                            0),
                            0),
                            0)
    
    # TODO: this sucks, bc it introduces an additional loop
    def a_array(l1, l2, pa, pb, cp, g):
        return jax.vmap(lambda iI : a_loop(iI, l1, l2, pa, pb, cp, g))(i_max_range)    

    imax = 2*l_max + 1
    rmax = jnp.floor(imax/2).astype(int)  + 1
    umax = rmax

    i_max_range = jnp.arange(imax)
    binomial_prefactor = get_binomial_prefactor(i_max_range)
    return a_array

def get_nuclear(l_max):

    def loop_body(i, j, k, lmn, rg):
        return jax.lax.cond( jnp.logical_and(jnp.logical_and(i <= lmn[0], j <= lmn[1]), k <= lmn[2]), lambda: gamma_fun(i+j+k , rg), lambda: 0.0)
        

    def loop(ax, ay, az, lmn, rg):
        return jax.lax.fori_loop(0, lim,
                    lambda i, vx: vx + jax.lax.fori_loop(0, lim,
                        lambda j, vy: vy + jax.lax.fori_loop(0, lim,
                            lambda k, vz: vz + ax[i] * ay[j] * az[k] * loop_body(i, j, k, lmn, rg),
                                0),
                                0),
                                0)
        
        
    def nuclear(alpha1, lmn1, pos1, alpha2, lmn2, pos2, nuc):
        gamma = alpha1 + alpha2
        p = (alpha1*pos1 + alpha2*pos2) / gamma
        rab2 = jnp.linalg.norm(pos1-pos2)**2
        rcp2 = jnp.linalg.norm(nuc-p)**2
        
        # TODO: this looks vectorizable
        vpa = p - pos1
        vpb = p - pos2
        vpn = p - nuc
        ax = a_array(lmn1[0], lmn2[0], vpa[0], vpb[0], vpn[0], gamma)
        ay = a_array(lmn1[1], lmn2[1], vpa[1], vpb[1], vpn[1], gamma)
        az = a_array(lmn1[2], lmn2[2], vpa[2], vpb[2], vpn[2], gamma)
        res = loop(ax, ay, az, lmn1+lmn2, rcp2*gamma)
        return -2.0 * jnp.pi / gamma * jnp.exp(-alpha1*alpha2*rab2/gamma) * res

    lim = 2*l_max+1
    a_array = get_a_array(l_max)
    
    return nuclear

def alpha(r):
    return 1/factorial(r)

def first_layer(l1, l2, a, b, g):
    
    return

def bb0(i, r, g):
    return factorial(i) / factorial(r) / factorial(i - 2*r) * jnp.pow(4*g,r-i)

def fb(i, l1, l2, p, a, b, r, g):
    return binomial_prefactor(i, l1, l2, p-a, p-b) * bb0(i, r, g)

def b_term(i, i1, i2, r1, r2, u, l1, l2, l3, l4, px, ax, bx, qx, cx, dx, g1, g2, delta):
    a, b = i1+i2-2*(r1+r2),u
    return (fb(i1,l1,l2,px,ax,bx,r1,g1)*
            jnp.pow(-1,i2) * fb(i2,l3,l4,qx,cx,dx,r2,g2)*
            jnp.pow(-1,u)* factorial(a) / factorial(b) / factorial(a - 2*b)*
            jnp.pow(qx-px,i1+i2-2*(r1+r2)-2*u)/
            jnp.pow(delta,i1+i2-2*(r1+r2)-u))

# TODO: this is restricted by i1 <= lmn1+lmn2, i2 <= lmn3+lmn4
def repulsion_beta(lmn_pos_dist_delta, angular_momenta_index_list):
    """Beta tensor for repulsion matrix element calculaton. This is a flat array, where beta[i] = convolution(b[0], b[1], b[2])[i]"""

    # TODO: compute b_tensor
    b_tensor = jax.vmap(f)(angular_momenta_index_list)

    # TODO: mask out illegal values
    # b_tensor = jnp.where(b_tensor, x, 0)

    # First convolve a and b
    beta_xy = jnp.convolve(b_tensor[0], b_tensor[1], mode='full')
    
    # Then convolve the result with c
    return jnp.convolve(b_tensor[2], beta_xy, mode='full')

# TODO: this is restricted by jax.lax.cond( jnp.logical_and(jnp.logical_and(i <= lmn[0], j <= lmn[1]), k <= lmn[2]), lambda: gamma_fun(i+j+k , rg), lambda: 0.0)
def repulsion_gamma(scaled_four_body_center, angular_momenta):
    """Gamma tensor for repulsion matrix element calculaton. This is a flat array, where gamma[i] = gamma_fun(i, scaled_four_body_center)
    
    Args:
        scaled_four_body_center : 
        angular_momenta : a range of total angular momenta such that I_max = 4*l_max

    Returns:
        jax.Array
    """
    
    return jax.vmap(lambda i : gamma_fun(i, scaled_four_body_center))(angular_momenta)

def tensor_element(orbs, angular_momenta, angular_momenta_index_list):
    # TODO: the loop is sum_{ijk} G[i+j+k] * B[0, i] * B[1, j] * B[2, k] = G @ B
    # where the B-array has elements B[I] = \sum_{i,j,k such that i+j+k = I} B[0, i] * B[1, j] * B[2, k]
    # thus, the B-array is a convolution of B[0], B[1], B[2] which we can FFT

    # TODO: pull this out of the loop and pack it into array
    rab2 = jnp.linalg.norm(pos1-pos2)**2
    rcd2 = jnp.linalg.norm(pos3-pos4)**2        
    gamma12 = alpha1 + alpha2
    p = (alpha1*pos1 + alpha2*pos2) / gamma12        
    gamma34 = alpha3 + alpha4        
    q = (alpha3*pos3 + alpha4*pos4) / gamma34
    rpq2 = jnp.linalg.norm(p-q)**2        
    delta = 0.25 * (1.0 / gamma12 + 1.0 / gamma34)

    scaled_four_body_center = XXX
    
    gamma = repulsion_gamma(scaled_four_body_center, angular_momenta)
    
    beta = repulsion_beta(lmn_pos_dist_delta)

    return prefac * (gamma @ beta)

def repulsion(orb1, orb2, orb3, orb4, angular_momenta, angular_momenta_index_list):
    """repulsion matrix element for cgf orbitals

       Args:
       orbitals 
       angular_momenta : a range of total angular momenta such that I_max = 4*l_max
       angular_momenta_index_list : a list of indices grouped by angular momentum for computing the B-tensor
    
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




