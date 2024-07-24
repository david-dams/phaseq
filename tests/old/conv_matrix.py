import jax
import jax.numpy as jnp

a = jnp.array( [4,1,2,1,3,11,12] )
b = jnp.arange(3) + 10

# manual convolution
res = [0.0] * (len(a) + len(b) - 1)
bprime = jnp.pad(b, (0, len(a)-1), mode = "constant") # pad mask to have size of res

# perform conv
for I in range(len(res)):
    for i in range(len(a)):
        res[I] += float(a[i] * bprime[I - i])

# comparison        
res = jnp.array(res)
comp = jnp.convolve(a, b)
print( (res - comp).sum() )

# matrix convolution
def reshape_to_matrix(x, n):
    N = x.shape[0]
    
    # Create a 2D array of indices
    row_indices = jnp.arange(N)[:, None]  # Shape: (N, 1)
    col_indices = jnp.arange(n)[None, :]  # Shape: (1, n)
    
    # Compute the shifted indices
    indices = row_indices - col_indices
    
    # Gather the elements using the shifted indices
    M = x[indices]
    
    return M

m = reshape_to_matrix(bprime, len(a))

print(comp - m @ a)

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

def reshape_to_matrix(x, n):
    N = x.shape[0]
    
    # Create a 2D array of indices
    row_indices = jnp.arange(N)[:, None]  # Shape: (N, 1)
    col_indices = jnp.arange(n)[None, :]  # Shape: (1, n)
    
    # Compute the shifted indices
    indices = row_indices - col_indices

    # Gather the elements using the shifted indices
    M = x[indices] * (row_indices <= 2 * col_indices)
    return M

m = reshape_to_matrix(bprime, len(a))

