import jax
import jax.numpy as jnp

## "restricted" matrix convolution ##
def reshape_restricted_conv(x, n):
    N = x.shape[0]
    
    # Create a 2D array of indices
    row_indices = jnp.arange(N)[:, None]  # Shape: (N, 1)
    col_indices = jnp.arange(n)[None, :]  # Shape: (1, n)
    
    # Compute the shifted indices
    indices = row_indices - col_indices

    # Gather the elements using the shifted indices
    M = x[indices] * (row_indices <= 2 * col_indices)
    return M

# restricted convolution
a = jnp.array( [4,1,2,1,3,11,12] )
b = jnp.arange(3) + 10

# manual convolution
res = [0.0] * (len(a) + len(b) - 1)
bprime = jnp.pad(b, (0, len(a)-1), mode = "constant") # pad mask to have size of res

# perform conv
for i in range(len(a)):        
    for j in range(len(b)):
        if j <= i:
            I = i + j
            res[I] += float(a[i] * b[j])
res = jnp.array(res)

m = reshape_restricted_conv(bprime, len(a))
print(m @ a - res)


## "restricted" cross correlation ##
def reshape_restricted_corr(x, n):
    N = x.shape[0]
    
    # Create a 2D array of indices
    row_indices = jnp.arange(N)[:, None]  # Shape: (N, 1)
    col_indices = jnp.arange(n)[None, :]  # Shape: (1, n)
    
    # Compute the shifted indices
    indices = row_indices - col_indices

    # Gather the elements using the shifted indices
    M = x[indices] 
    return M

a = jnp.array( [4,1,2,1,3,11,12] )
b = jnp.arange(3) + 10

# only positive indices => resulting array will have the same length as a
f = [0] * a.size
for i in range(a.size):
    for j in range(b.size):
        if i >= j:
            f[i - j] += float(a[i] * b[j])

# convolution            
F = jnp.convolve(a, b[::-1])

bprime = jnp.pad(b[::-1], (0, len(a)-1), mode = "constant") # pad mask to have size of res
m = reshape_restricted_corr(bprime, len(a))
print(m @ a - F)

$\sum\limits_{i_1 + i_2 - 2(r_1 + r_2) - u = I} a_{i_1, r_1} b_{i_2, r_2} M_{i_1 i_2 r_1 r_2 u} = \sum\limits_{k - u = I} c_k M_{k, u} = \sum\limits_{k} c_{k} M_{k, I+k}$
# jnp.sum(c @ M, axis = -1)

