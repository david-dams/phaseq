# transformations from nested loops to convolutions and masked dot products #
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def array_to_matrix(x):
    N = x.shape[0]
    
    # Create a 2D array of indices
    row_indices = jnp.arange(N)[:, None]  # Shape: (N, 1)
    col_indices = jnp.arange(N)[None, :]  # Shape: (1, n)
    
    # Compute the shifted indices
    indices = col_indices - row_indices

    # Gather the elements using the shifted indices, masking out "wrap-around" indices
    M = x[indices] * (indices >= 0)
    return M

def test_restricted_matrix_cross_correlation(tolerance = 1e-12):
    """restricted matrix cross-correlations of type
    $A_I = \sum\limits_{K - u = I} f(K) \cdot d(K, u)$
    """

    def f(K):
        return 1/(K+1)

    def d(K, u):
        return K * u

    # explicit sum
    kmax= 12
    ref = [0.0] * kmax
    for k in range(kmax):
        for u in range(k + 1):
            I = K - u
            if I >= 0:
                ref[I] += f(K) * d(K, u)
    ref = jnp.array(ref)
 
    # matrix magic
    ds = jnp.array( [d(K, u) for k in range(kmax) for u in range])
    fs = jnp.array( [f(K) for k in range()])

    res = ds @ fs

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
                
                
