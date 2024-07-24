# tests for common functions
import numpy as np
from scipy.special import binom, factorial

from phaseq import *

def binomial_ref(a, b):
    if (a < 0) or (b < 0) or (a-b < 0):
        return 1.0
    return factorial(a) / (factorial(b) * factorial(a-b))

def binomial_prefactor_ref(s, l1, l2, x1, x2):
    sum = 0.0
    for t in range(s+1):
        if ((s-l1 <= t) and (t <= l2)):
            sum += (binomial_ref(l2, t) * binomial_ref(l1, s-t) *
                   np.power(x2, l2-t) * np.power(x1, l1-(s-t)))
    return sum

def test_binomial_prefactor(tolerance = 1e-10):    
    a1, l1, m1, n1, x1, y1, z1 =  0.2, 4, 3, 0, 0.2, 0.3, 0.1
    a2, l2, m2, n2, x2, y2, z2 =  0.1, 2, 1, 3, 10., 0.1, 0.5    
    gaussian1 = jnp.array( [x1, y1, z1, l1, m1, n1, a1] )
    gaussian2 = jnp.array( [x2, y2, z2, l2, m2, n2, a2] )        
    l_max = max(l1, m1, n1, l2, m2, n2) + 1
    l_arr = jnp.arange(l_max)
    t_arr = jnp.arange(2*l_max+1)   
    bf = binomial_prefactor(2*l_arr, gaussian1, gaussian2, t_arr)
    
    for j in range(3):
        l1 = float(gaussian1[3:6][j])
        l2 = float(gaussian2[3:6][j])
        x1 = gaussian1[:3][j]
        x2 = gaussian2[:3][j]
        for i in range(1 + int(np.floor(0.5 * (l1 + l2)))):
            bf_ref = binomial_prefactor_ref(2*i, l1, l2, x1, x2)
            assert jnp.abs(bf[i, j] - bf_ref) < tolerance
            
if __name__ == '__main__':
    test_binomial_prefactor()
