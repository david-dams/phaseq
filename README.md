# PhaseQ 

Electron matrix elements as differentiable compute graphs.

## Matrix Element Graphs

A general electronic Hamiltonian can be expressed (denoting creators / annihilators by capital / small letters) as

$H = T^{ab} Ab + V_{ab} Ab + U_{abcd} ABcd$

where $T,V, U$ are the kinetic, nuclear and interaction matrix representations.


The non-orthonormality of the basis can be taken into account via

$\{A,b\} = S_{ab}$

The following non-recursive expressions for the matrix elements occuring in the expression above have been derived by [Takaeta et al.](https://csclub.uwaterloo.ca/~pbarfuss/jpsj.21.2313.pdf)

TBD

In preparation of the following sections, we introduce the following definitions for operations on a sequence $a$ of length $I$

$\text{reverse} : a \to b, b_i = a_{I-i}$
$\text{inflate} : a \to b, b_{2i} = a_{i}, b_{2i+1} = 0$
$\text{Conv}_3 : a,b,c \rightarrow \text{Conv}[\text{Conv}[a,b],c]$

Additionally, we denote for a two-dimensional sequence $A$ with $A_i$ the $j$ indexed subsequence of all elements $A_{ij}$ with $i$ fixed. 

We first focus on the expressions for primitive Gaussians.

### Overlap
The overlap integrals can be recast as follows

$a \cdot \prod_i b(l_1, l_2, \vec{d}_{AP}, \vec{d}_{BP}, i) c$

where 

$a = \frac{\pi}{\gamma}^{3/2} e^{-\alpha_1 \alpha_2 \vec{d}_{AB}^2 / \gamma}$
$b = f_{2i}(l_1, l_2,\vec{d}_{i, AP}, \vec{d}_{i, BP} )$
$c_i = \frac{(2i-1)!!}{(2 \gamma)^i}$

### Kinetic
$\alpha_2 \cdot (2(l_2+m_2+n_2) + 3)S(l_2, m_2, n_2) - 2 \alpha_2^2(S(l_2+2,m_2,n_2) + S(l_2,m_2+2,n_2) + S(l_2,m_2,n_2+2)) - \frac{1}{2}( l_2(l_2-1)S(l_2-2, m_2, n_2) + m_2(m_2-2)S(l_2, m_2-2, n_2) + n_2(n_2 - 1)S(l_2, m_2, n_2-2))$

### Nuclear
The first group of nesting levels can be written as

$\sum_{ijk} A_i B_j C_k F_{i+j+k} = \sum_I F_I \sum_{i+j+k=I} A_i B_j C_k = \sum_I F_I \text{Conv}_3[A,B,C]_I$

where $\text{Conv}_3$ denotes the triple convolution operator.

The second group of nesting levels can be transformed upon redefining $d_{I, u} = d_{I-u}$ as follows

$A_I = \sum_{i-2r-u = I} a_i b_r c_u d_{i-2r-2u} = \text{Conv}_3[a, b', c'_I ]_I$

where $b'= \text{reverse}[\text{inflate}[b]], c'_I = \text{reverse}[c \odot d_{I}]$ and

$a_i = i!$
$b_r(\epsilon) = \frac{\epsilon^r}{r!}$
$\epsilon = \gamma / 4$
$c_u(\epsilon) = \frac{(-1)^u \epsilon^u}{u!}$
$d_{i-2r-2u}(p) = \frac{p^{i-2r-2u}}{(i-2r-2u)!}$

### Interaction

The first group of nesting levels can be written as

$\sum_{ijk} A_i B_j C_k F_{i+j+k} = \sum_I F_I \sum_{i+j+k=I} A_i B_j C_k = \sum_I F_I \text{Conv}_3[A,B,C]_I$

where $\text{Conv}_3$ denotes the triple convolution operator.

The second group of nesting levels can be transformed to be
   
$A_I = \sum_{l+m+n = I} a_l b_m c_{l+m+n,n} = \text{Conv}_3[a,b,c_I]_{I}$

where

$a_l = a_l(l_1, l_2, pa, pb, g_1)$
$b_m = (-1)^m a_m(l_3, l_4, qc, qd, g_2)$
$c_{l+m+n,n}(p) = \frac{(l+m)! p^{l+m+2n}(-1)^{-n}}{n!(l+m+2n)! \delta^{l+m+n}}$

Finally, the third group of nesting levels can be recast to

$a_I = \sum_{i-2j = I} \alpha_i \beta_j \gamma_{i-2j} = \gamma_{I} \sum_{i-2j=I} \alpha_{i} \beta_{j} = \gamma_{I} \text{Conv}[\alpha, \beta']_I$

where

$\alpha_i = \frac{1}{i!}$
$\beta' = \text{reverse}[\text{inflate}[\beta]]$
$\beta_j(l_1, l_2, a, b) = j! f_j(l_1, l_2, a, b)$
$\gamma_{i-2j}(g) = \frac{1}{(i-2j)! (4g)^{i-2j}}$
$f_j(l_1, l_2, a, b) = \partial^j_x (a+x)^{l_1} (a+x)^{l_2} \vert_{x=0}$

The binomial prefactor $f_j$ may be expressed as a slightly awkward masked dot product

$f_s(l_1, l_2, a, b) = \sum\limits_{s-l_1 \leq t \leq l_2, t \leq s + 1} \lambda_{t}(l_1, a) \lambda_{s - t}(l_2, b) = \text{Conv}[\zeta, \sigma]_s$
