# transformations from nested loops to convolutions and masked dot products #
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def kernel_to_matrix(kernel, imax):
    """Reshapes the N-dim array `kernel` into a N x N matrix
    
    out = kernel @ signal

    Is the output array of the discrete convolution.
    """
    # conditionally skip for odd total angular momnentum
    rmax = imax // 2 + 1 if (imax % 2 == 1) else imax // 2
    kernel = kernel[:rmax]
    
    # "inflate" g such that g[2*r] = g(r)
    kernel = jnp.insert(arr = kernel, obj = jnp.arange(0, kernel.size) + 1, values = 0)

    if imax % 2 == 1:
        kernel = kernel[:-1]
            
    N = kernel.shape[0]
    
    # Create a 2D array of indices
    row_indices = jnp.arange(N)[:, None]  # Shape: (N, 1)
    col_indices = jnp.arange(N)[None, :]  # Shape: (1, n)
    
    # Compute the shifted indices
    indices = col_indices - row_indices

    # reshape g such that gs[L, i] = gs[i - L]
    # Gather the elements using the shifted indices, masking out "wrap-around" indices
    M = kernel[indices] * (indices >= 0)
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
    imax= 12
    ref = [0.0] * imax
    for i in range(imax):
        for r in range(i//2 + 1):
            L = i - 2*r
            if L >= 0:
                ref[L] += f(i) * g(r) * h(L)
    ref = jnp.array(ref)

    # matrix magic

    # first, "inflate" g such that g[2*r] = g(r)
    rmax = imax // 2 + 1 if (imax % 2 == 1) else imax // 2   
    gs = jnp.array([g(i) for i in range(rmax)])
    gs = jnp.insert(arr = gs, obj = jnp.arange(0, gs.size) + 1, values = 0)
    if imax % 2 == 1:
        gs = gs[:-1]

    # compute fs
    fs = jnp.array([f(i) for i in range(imax)])
    
    # compute hs
    hs = jnp.array([h(i) for i in range(len(gs))])

    # reshape g such that gs[L, i] = gs[i - L]
    gs_old = gs
    gs = array_to_matrix(gs)
    
    # compute masked cross-correlation
    res = gs @ fs * hs

    # compute difference
    delta = res - ref    
    print(delta)
    assert jnp.abs(delta).sum() <= tolerance

if __name__ == '__main__':
    test_restricted_cross_correlation()
                
                
