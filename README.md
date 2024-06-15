# PhaseQ 

Electron matrix elements as differentiable compute graphs.

## Matrix Element Graphs

Hamiltonian

TBD

The following non-recursive expressions for the matrix elements occuring in the expression above have been derived by [Takaeta et al.](https://csclub.uwaterloo.ca/~pbarfuss/jpsj.21.2313.pdf)

TBD

These can be implemented as a chain of convolutions and masked dot products as follows:

### Overlap

### Kinetic

### Nuclear

### Interaction

The first group of nesting levels can be written as

$\sum_{ijk} A_i B_j C_k F_{i+j+k} = \sum_I F_I \sum_{i+j+k=I} A_i B_j C_k = \sum_I F_I \text{Conv}_3[A,B,C]_I$

where $\text{Conv}_3$ denotes the triple convolution operator.

The second group of nesting levels can be transformed to be
   
$A_I = \sum_{l+m+n = I} a_l b_m c_{l+m,n} = \sum_{r+n=I} c_{r,n} \sum_{l+m=r} a_l b_m = \sum_{r+n=I} c_{r,n} \text{Conv}[a,b]_r = \sum_{n \leq I} c_{I-n, n} \text{Conv}[a,b]_{I-n}$

where

$a_l = a_l(l_1, l_2, pa, pb, g_1)$
$b_m = (-1)^m a_m(l_3, l_4, qc, qd, g_2)$
$c_{l+m,n}(p) = \frac{(l+m)! p^{l+m-2n}(-1)^n}{n!(l+m-2n)! \delta^{l+m-n}}$

$A_I$ is thus given by the sum of the antidiagonal of the $I$ -th submatrix of $M = c * \text{diag}{\text{Conv}[a,b]}$, where $*$ denotes a row-wise product.

Finally, the third group of nesting levels can be recast to

$a_I = \sum_{i+j = I} \alpha_i \beta_j \gamma_{i+j} = \gamma_{I} \sum_{i+j=I} \alpha_{i} \beta_{j} = \gamma_{I} \text{Conv}[\alpha, \beta]_I$

where

$\alpha_i = \frac{1}{i!}$
$\beta_j(l_1, l_2, a, b) = j! f_j(l_1, l_2, a, b)$
$\gamma_{i+j}(g) = \frac{1}{(i-2j)! (4g)^{i-2j}}$
$f_j(l_1, l_2, a, b) = \partial^j_x (a+x)^{l_1} (a+x)^{l_2} \vert_{x=0}$





