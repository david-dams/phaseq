# python implementation of functions defined in reference
import numpy as np
from scipy.special import binom

def binomial_prefactor_ref(s, l1, l2, x1, x2):
    sum = 0.0
    for t in range(int(max(0, s-l2)), int(min(s, l1)+1)):
        sum += (binom(l1, t) * binom(l2, s-t) *
                np.power(x1, l1-t) * np.power(x2, l2-(s-t)))
    print(sum)
    return sum
    
def double_factorial(n):
    if n <= 0:
        return 1
    else:
        return n * double_factorial(n-2)

def overlap_1D(l1, l2, x1, x2, gamma):
    sum = 0.0
    for i in range(1 + int(np.floor(0.5 * (l1 + l2)))):
        sum += (binomial_prefactor_ref(2*i, l1, l2, x1, x2) *
                (1 if i == 0 else double_factorial(2 * i - 1)) /
                np.power(2 * gamma, i))
    return sum

def overlap_ref(alpha1, l1, m1, n1, a, alpha2, l2, m2, n2, b):
    def gaussian_product_center(alpha1, a, alpha2, b):
        return (alpha1 * a + alpha2 * b) / (alpha1 + alpha2)
    
    rab2 = np.dot(a-b, a-b)
    gamma = alpha1 + alpha2
    p = gaussian_product_center(alpha1, a, alpha2, b)

    pre = np.power(np.pi / gamma, 1.5) * np.exp(-alpha1 * alpha2 * rab2 / gamma)
    wx = overlap_1D(l1, l2, p[0]-a[0], p[0]-b[0], gamma)
    wy = overlap_1D(m1, m2, p[1]-a[1], p[1]-b[1], gamma)
    wz = overlap_1D(n1, n2, p[2]-a[2], p[2]-b[2], gamma)

    return pre * wx * wy * wz
