import jax
import jax.numpy as jnp

### triple accumulated loop is triple convolution ###
# $\sum_{i+j+k=I} a_i b_j c_k = Conv[[Conv[a,b], c]_I$
a = jnp.arange(5) + 3
b = jnp.arange(4) + 7
c = jnp.arange(3) + 9
f = [0] * (a.size + b.size + c.size - 2)
for i in range(a.size):
    for j in range(b.size):
        for u in range(c.size):
            f[i+j+u] += float(a[i] * b[j] * c[u])
F = jnp.convolve(jnp.convolve(a,b),c)
assert (F - jnp.array(f)).sum() == 0


### double loop accumulated over positive indices ###
# this is "half" an autocorrelation
# $r[I = i-j \geq 0] = \sum_{i-j \geq 0} a_i b_j = Conv[a, Rev[b]]_{I \geq 0}$
# this means we need the "positive part" of the result array r, skipping the strictly positive index range of b
# i.e. r[(b.size-1):]
a = jnp.arange(11) + 3
b = jnp.arange(10) + 7

# only positive indices => resulting array will have the same length as a
f = [0] * a.size
for i in range(a.size):
    for j in range(b.size):
        if i >= j:
            f[i - j] += float(a[i] * b[j])

# convolution            
F = jnp.convolve(a, b[::-1])

# skip indices
assert (F[(b.size-1):]  - jnp.array(f)).sum() == 0

### triple loop, one positive, two negative indices ###
# $\sum_{i-j-k = I \geq 0} a_i b_j c_k = \sum_{i+j' = I \geq 0} a_i b'_j = Conv[a, b']_{I \geq 0}$
# b' = Conv[b,c]
a = jnp.arange(5) + 3
b = jnp.arange(4) + 7
c = jnp.arange(3) + 11

# only positive indices => a sets the size
f = [0] * a.size
for i in range(a.size):
    for j in range(b.size):
        for k in range(c.size):
            if i >= j + k:
                f[i - j - k] += float(a[i] * b[j] * c[k])

# regular convolution                
x = jnp.convolve(b, c)

# flip for negative indices cross-correlation
F = jnp.convolve(a, x[::-1])

# cut the correct index range by skipping the strictly positive indices of x
assert (F[(x.size-1):]  - jnp.array(f)).sum() == 0


### triple loop, one positive, one negative index, one negative shifted / multiplied index ###
# $\sum_{i-2j-k = I \geq 0} a_i b_j c_k = \sum_{i+j' \geq 0} a b' = Conv[a,b']$
# $b' = Conv[Inflate[b], c]$
a = jnp.arange(11) + 3
b = jnp.arange(10) + 7
c = jnp.arange(5) + 11

# only positive indices => a sets the size
f = [0] * a.size
for i in range(a.size):
    for j in range(b.size):
        for k in range(c.size):
            if i >= 2*j + k:
                f[i - 2*j - k] += float(a[i] * b[j] * c[k])

# convolution with inflated array                
b_inflated = jnp.insert(arr = b, obj = jnp.arange(0, b.size) + 1, values = 0)                
x = jnp.convolve(b_inflated, c)

# flip for negative indices cross-correlation
F = jnp.convolve(a, x[::-1])

# cut the correct index range by skipping the strictly positive indices of x
assert (F[(x.size-1):] - jnp.array(f)).sum() == 0


### triple loop times permutation symmetric function ###
# $\sum_{i-2j-k = I \geq 0} a_i b_j c_k d_{i+j+k} = d_I \sum_{i+j' = I \geq 0} a b' = d_I * Conv[a,b']_I$
# $b' = Conv[Inflate[b], c]$
a = jnp.arange(11) + 3
b = jnp.arange(10) + 7
c = jnp.arange(5) + 11

def symmetric(I):
    return I**2

# only positive indices => a sets the size
f = [0] * a.size
for i in range(a.size):
    for j in range(b.size):
        for k in range(c.size):
            if i >= 2*j + k:
                f[i - 2*j - k] += float(a[i] * b[j] * c[k]) * symmetric(i - 2*j - k)

# convolution with inflated array                
b_inflated = jnp.insert(arr = b, obj = jnp.arange(0, b.size) + 1, values = 0)                
x = jnp.convolve(b_inflated, c)

# flip for negative indices cross-correlation
F = jnp.convolve(a, x[::-1])

# compute function on the corresponding index range => length of a
func_vals = symmetric(jnp.arange(a.size))

# cut the correct index range by skipping the strictly positive indices of x
assert (F[(x.size-1):] * func_vals  - jnp.array(f)).sum() == 0


### triple loop over triple loop with permutation symmetric function ###
# $\sum_{I} \sum_{i+j+k = I} a_i b_j c_k d_{i+j+k} = \sum_I d_I Conv[Conv[a,b],c]_I$
# where a_i is given by triple loop result
def symmetric(I):
    return 1/(I**2+1)

def inner_loop(a,b,c):
    # only positive indices => a sets the size
    f = [0] * a.size
    for i in range(a.size):
        for j in range(b.size):
            for k in range(c.size):
                if i >= 2*j + k:
                    f[i - 2*j - k] += float(a[i] * b[j] * c[k])
    return jnp.array(f)

def outer_loop(a,b,c):
    # total size set by all arrays
    val = 0.
    for i in range(a.size):
        for j in range(b.size):
            for k in range(c.size):
                val += float(a[i] * b[j] * c[k]) * symmetric(i+j+k)
    return val

# outer and inner loop repeat three times => we need 3 * 3 = 9
a1 = jnp.linspace(0,1,11) + 3
b1 = jnp.arange(0,3,10) + 7
c1 = jnp.arange(0,2,5) + 11

a2 = jnp.arange(0,2,11) + 3
b2 = jnp.arange(0,1,10) + 7
c2 = jnp.arange(0,3,5) + 11

a3 = jnp.arange(0,4,11) + 3
b3 = jnp.arange(0,1,10) + 7
c3 = jnp.arange(0,2,5) + 11

a = inner_loop(a1, b1, c1)
b = inner_loop(a2, b2, c2)
c = inner_loop(a3, b3, c3)

val = outer_loop(a,b,c)

# inner loops given by three convolutions
def inner_loop_conv(a,b,c):
    # convolution with inflated array
    b_inflated = jnp.insert(arr = b, obj = jnp.arange(0, b.size) + 1, values = 0)
    x = jnp.convolve(b_inflated, c)
    # cut the correct index range by skipping the strictly positive indices of x
    F = jnp.convolve(a, x[::-1])[(x.size-1):]    
    return F

# outer loop given by convolution and product
def outer_loop_conv(a,b,c):
    x = jnp.convolve(a,b)
    F = jnp.convolve(x, c)
    func_vals = symmetric(jnp.arange(F.size))
    return F @ func_vals

a_conv = inner_loop_conv(a1, b1, c1)
b_conv = inner_loop_conv(a2, b2, c2)
c_conv = inner_loop_conv(a3, b3, c3)

val_conv = outer_loop_conv(a_conv, b_conv, c_conv)

# cut the correct index range by skipping the strictly positive indices of x
assert jnp.abs(val - val_conv) < 1e-8
