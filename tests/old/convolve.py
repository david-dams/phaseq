import jax
import jax.numpy as jnp

# corr(a, b) = \sum_{j} a_j b_{j+k} = conv(a, b[::-1])
# \sum_{i-j=k} a_i b_j = \sum_{j} a_{k+i} b_j = Conv[a[::-1],b]

# a = jnp.arange(5) + 3
# b = jnp.arange(3) + 9
# x = [0]*( (a.size+2*b.size) - 1)
# for i in range(a.size):
#     for j in range(b.size):
#         if -i + 2*j >= 0:
#             x[-i + 2*j] += float(a[i] * b[j])        
        
# b = jnp.insert(arr = b, obj = jnp.arange(0, b.size)+1, values = 0)
# c = jnp.convolve(a[::-1], b)            
# # c = jnp.convolve(a, b)            
# print(x)
# print(c[(a.size-1):])


# a = jnp.arange(5) + 3
# b = jnp.arange(3) + 9
# x = [0]*( (a.size+2*b.size) - 1)
# for i in range(a.size):
#     for j in range(b.size):
#         x[i + 2*j] += float(a[i] * b[j])        
        
# b = jnp.insert(arr = b, obj = jnp.arange(0, b.size)+1, values = 0)
# c = jnp.convolve(a[::-1], b)            
# # c = jnp.convolve(a, b)            
# print(x)
# print(c)


# a = jnp.arange(5) + 3
# b = jnp.arange(3) + 9
# c = jnp.arange(3) - 10

# x = [0]*( (a.size+2*b.size) - 2)
# for i in range(a.size):
#     for j in range(b.size):
#         for k in range(c.size):    
#             x[i + j + k] += float(a[i] * b[j] * c[k])
            
# d = jnp.convolve(jnp.convolve(a, b), c)
# print(x)
# print(d)




# a = jnp.arange(5) + 3
# b = jnp.arange(3) + 9
# c = jnp.convolve(a, b)            
# total = a.size + b.size - 1
# a_pad = jnp.zeros(total - a.size)
# a = jnp.concatenate( [a, a_pad] )
# b_pad = jnp.zeros(total - b.size)
# b = jnp.concatenate( [b, b_pad] )
# x = [0] * total
# for n in range(total):
#     for m in range(total):
#         if n - m  >= 0:
#             x[n] += float(a[m] * b[n-m])
                    
# print(x)
# print(c)


# a = jnp.arange(5) + 3
# b = jnp.arange(3) + 9
# c = jnp.convolve(a, b)            
# total = a.size + b.size - 1
# a_pad = jnp.zeros(total - a.size)
# a = jnp.concatenate( [a, a_pad] )
# b_pad = jnp.zeros(total - b.size)
# b = jnp.concatenate( [b, b_pad] )
# x = [0] * total
# for n in range(total):
#     for m in range(total):
#         if n - m  >= 0:
#             x[n] += float(a[m] * b[n-m])
                    
# print(x)
# print(c)


a = jnp.arange(5) + 3
b = jnp.arange(5) + 9
# a = jnp.ones_like(a)
# b = jnp.ones_like(b)
c1 = jnp.convolve(a[::-1], b, 'full')
c2 = jnp.convolve(a, b[::-1], 'full')            
total = a.size + b.size - 1
# a_pad = jnp.zeros(total - a.size)
# a = jnp.concatenate( [a, a_pad] )
# b_pad = jnp.zeros(total - b.size)
# b = jnp.concatenate( [b, b_pad] )
x = [0] * total
for k in range(total):
    for i in range(a.size):
        for j in range(b.size):
            if k == i - j:
                x[k] += float(a[i] * b[j])

                
print(x)
# print(jnp.convolve(a, b, 'full'))
print(c1)
print(c2)

a = jnp.arange(5) + 3
b = jnp.arange(5) + 9
c = jnp.arange(5) + 9
# a = jnp.ones_like(a)
# b = jnp.ones_like(b)
c1 = jnp.convolve(a[::-1], b, 'full')
c2 = jnp.convolve(a, b[::-1], 'full')            
total = a.size + b.size - 1
# a_pad = jnp.zeros(total - a.size)
# a = jnp.concatenate( [a, a_pad] )
# b_pad = jnp.zeros(total - b.size)
# b = jnp.concatenate( [b, b_pad] )
x = [0] * total*2
for k in range(total):
    for i in range(a.size):
        for j in range(b.size):
            for l in range(c.size):
                if k == i - j - l:
                    x[k] += float(a[i] * b[j] * c[l])

                
print(x)
# print(jnp.convolve(a, b, 'full'))
print(c1)
print(c2)


lmax = 3
f = []
for i in range(lmax+1):
    for j in range( int(i/2)+1 ):
        for u in range( int((i -2*j)/2)+1 ):
            f.append( i - 2*j - u )

import matplotlib.pyplot as plt

f = [ i - 2*j - u for i in range(10) for j in range( i // 2 + 1) for u in range( (i - 2*j) // 2 + 1) ]


# indices in the loop
# lmax = 3
# f = []
# for i in range(lmax+1):
#     for j in range( int(i/2)+1 ):
#         for u in range( int((i -2*j)/2)+1 ):
#             f.append( i - 2*j - u )
