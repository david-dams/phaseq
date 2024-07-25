# transformations from nested loops to convolutions and masked dot products #
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def kernel_to_matrix(kernel, imax, imax_lim):
    """Helper function for vectorized evaluation of certain "restricted cross-correlations" occuring in nuclear and interaction matrix elements.

    Args:
      kernel : N- array containing kernel elements of the restricted cross-correlation
      imax   : int, N

    Allows vectorized evaluation of an expression like $T(I) =  \sum\limits_{i - 2r = I, r \leq i/2} a(i) b(r)$ where i/2 denotes integer division as 
    
    T = B @ a
    
    where B is a matrix. This occurs in evaluation nuclear and interaction matrix elements.

    This works as follows:

    1. Inflate the array b' of b-values such that b'[2r] = b[r] and b[2r+1] = 0.
    2. Then, since $\sum\limits_{i - j = I, j \leq i} a[i] b'[j] = \sum\limits_{i, I \geq 0} a[i] b'[i - I]$ define a matrix B such that B[I, i] = b'[i - I].
    """
    # conditionally skip for odd total angular momnentum
    rmax = imax // 2 # TODO: move out of loop for fixed shape?
    rmax_lim = imax_lim // 2 

    # zero-out elements
    kernel *= (jnp.arange(imax) <= rmax_lim)
    
    # "inflate" g such that g[2*r] = g(r)
    kernel = jnp.insert(arr = kernel[:rmax], obj = jnp.arange(rmax) + 1, values = 0)
                
    # Create a 2D array of indices
    row_indices = jnp.arange(imax)[:, None]  # Shape: (N, 1)
    col_indices = jnp.arange(imax)[None, :]  # Shape: (1, n)
    
    # Compute the shifted indices
    indices = col_indices - row_indices

    # reshape g such that gs[L, i] = gs[i - L]
    # Gather the elements using the shifted indices, masking out "wrap-around" indices
    M = kernel[indices] * (indices >= 0)
    M = M.at[:, (imax_lim-imax):].set(0)

    return M

def test_restricted_cross_correlation(tolerance = 1e-12):
    """restricted cross-correlations of type
    $c_L = \sum\limits_{i - 2r = L, r \leq i/2} f(i) \cdot g(r) \cdot h(i - 2r) = h(L) \cdot \sum_{i} G_{L, i} f_i$
    Where $G$ is the "inflated" matrix form of the array $G_{2r} = g(2r)$ such that $G_{L, i} = G_{i - L}$
    """

    def f(i):
        return (i + 3)**2
    
    def g(r):
        return 3**r

    def h(L):
        return 1 / (L + 1)

    # explicit sum
    imax = 14
    ref = [0.0] * imax
    imax_lim = 4
    for i in range(imax_lim):
        for r in range(i//2 + 1):
            L = i - 2*r
            if L >= 0:
                ref[L] += f(i) * g(r) * h(L)
    ref = jnp.array(ref)

    # matrix magic

    # first, "inflate" g such that g[2*r] = g(r)
    gs = jnp.array([g(i) for i in range(imax)])

    # compute fs
    fs = jnp.array([f(i) for i in range(imax)])
    
    # compute hs
    hs = jnp.array([h(i) for i in range(imax)])

    # reshape g such that gs[L, i] = gs[i - L]
    mat = kernel_to_matrix(gs, imax, imax_lim)
    
    # compute masked cross-correlation
    res = mat @ fs * hs

    # compute difference
    delta = res - ref    
    # print(res, ref, delta)
    assert jnp.abs(delta).sum() <= tolerance

if __name__ == '__main__':
    test_restricted_cross_correlation()
                
                
