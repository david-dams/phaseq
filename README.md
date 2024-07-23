# PhaseQ 

Electron matrix elements as differentiable compute graphs.

## Matrix Element Graphs

A general electronic Hamiltonian can be expressed (denoting creators / annihilators by capital / small letters) as

$H = T_{ab} Ab + V_{ab} Ab + U_{abcd} ABcd$

where $T,V, U$ are the kinetic, nuclear and interaction matrix representations.


The non-orthonormality of the basis can be taken into account via

$\{A,b\} = S_{ab}$

The following non-recursive expressions for the matrix elements occuring in the expression above have been derived by [Takaeta et al.](https://csclub.uwaterloo.ca/~pbarfuss/jpsj.21.2313.pdf)

TBD

Where 

$f_j(l_1, l_2, a, b) = \partial^j_x (a+x)^{l_1} (a+x)^{l_2} \vert_{x=0}$

In preparation of the following sections, we introduce the following definitions for operations on a sequence $a$ of length $I$

$\text{reverse} : a \to b, b_i = a_{I-i}$
$\text{inflate} : a \to b, b_{2i} = a_{i}, b_{2i+1} = 0$
$\text{Conv}_3 : a,b,c \rightarrow \text{Conv}[\text{Conv}[a,b],c]$

Additionally, we denote for a two-dimensional sequence $A$ with $A_i$ the $j$ indexed subsequence of all elements $A_{ij}$ with $i$ fixed. 

We first focus on the expressions for primitive Gaussians.

### Overlap
The overlap integrals can be recast as follows

$a \cdot \prod_i \sum_I b_I(l_1, l_2, \vec{d}_{AP}, \vec{d}_{BP}, i) c_I$

where 

$
\begin{align*}
a &= \frac{\pi}{\gamma}^{3/2} e^{-\alpha_1 \alpha_2 \vec{d}_{AB}^2 / \gamma} \\
b &= f_{2i}(l_1, l_2,\vec{d}_{i, AP}, \vec{d}_{i, BP} ) \\
c_i &= \frac{(2i-1)!!}{(2 \gamma)^i}
\end{align*}
$

### Kinetic
$\alpha_2 \cdot (2(l_2+m_2+n_2) + 3)S(l_2, m_2, n_2) - 2 \alpha_2^2(S(l_2+2,m_2,n_2) + S(l_2,m_2+2,n_2) + S(l_2,m_2,n_2+2)) - \frac{1}{2}( l_2(l_2-1)S(l_2-2, m_2, n_2) + m_2(m_2-2)S(l_2, m_2-2, n_2) + n_2(n_2 - 1)S(l_2, m_2, n_2-2))$

### Nuclear
The first group of nesting levels can be written as

$\sum_{ijk} A_i B_j C_k F_{i+j+k} = \sum_I F_I \sum_{i+j+k=I} A_i B_j C_k = \sum_I F_I \text{Conv}_3[A,B,C]_I$

where $\text{Conv}_3$ denotes the triple convolution operator. The summation over $I$ ranges from $0$ to $l_1 + l_2 + m_1 + m_2 + n_1 + n_2 - 2$.

The second group of nesting levels can be transformed upon redefining $d_{I, u} = d_{I-u}$ as follows

$A_I = \sum_{i-2r-u = I} a_i b_r c_u d_{i-2r-2u} = \text{Conv}_3[a, b', c'_I ]_I$

where $b'= \text{reverse}[\text{inflate}[b]], c'_I = \text{reverse}[c \odot d_{I}]$ and

$
\begin{align*}
a_i &= i! (-1)^if_i \\
b_r(\epsilon) &= \frac{\epsilon^r}{r!} \\
\epsilon &= \gamma / 4 \\
c_u(\epsilon) &= \frac{(-1)^u \epsilon^u}{u!} \\
d_{i-2r-2u}(p) &= \frac{p^{i-2r-2u}}{(i-2r-2u)!}
\end{align*}
$

### Interaction

The first group of nesting levels can be written as

$\sum_{ijk} A_i B_j C_k F_{i+j+k} = \sum_I F_I \sum_{i+j+k=I} A_i B_j C_k = \sum_I F_I \text{Conv}_3[A,B,C]_I$

where $\text{Conv}_3$ denotes the triple convolution operator and $A, B, C$ correspond to $x, y, z$ quantities and $F_I = F(I, \overline{PQ}^2 / (\gamma_1 + \gamma_2))$

The second group of nesting levels reads as

$A_I = \sum\limits_{r_1 \leq i_1 / 2, r_2 \leq i_2 / 2, u  \leq (i_1 + i_2)/2 - r_1 - r_2}^{i_1 + i_2 - 2(r_1 + r_2) - u = I} a_{i_1, r_1} b_{i_2, r_2} d_{I + u, u}$, 

where 

$
\begin{align}
a_{i_1, r_1} &= \frac{f_{i_1} i_1!}{r_1! (i_1 - 2 r_1)! (4 \gamma_1)^{i_1 - r_1}} \\
b_{i_2, r_2} &= \frac{(-)^{i_2} f_{i_2} i_2!}{r_2! (i_2 - 2 r_2)! (4 \gamma_2)^{i_2 - r_2}} \\
d_{I + u, u} &= \frac{ (I + u)! (-)^u p_x^{I - u}}{u!(I-u)!\delta^{I}}
\end{align}
$

where $f_{i_1} = f(i_1, \overline{PA}_x, \overline{PB}_x), f_{i_2} = f(i_2, \overline{QC}_x, \overline{QD}_x)$ refers to the
binomial prefactors of the gaussian pairs with respect to their centers and $p_x$  is the center-center distance $Q-P$ and $\delta = \frac{1}{4 \gamma_1} + \frac{1}{4 \gamma_2}$. We now rewrite

$
\begin{align}
a_L &= \frac{1}{L!}\sum\limits_{r_1 \leq i_1 / 2}^{i_1 - 2r_1 = L} \frac{f_{i_1} i_1!}{(4 \gamma_1)^{i_1}} \frac{(4 \gamma_1)^{r_1}}{r_1!}\\
b_M &= \frac{1}{M!}\sum\limits_{r_2 \leq i_2 / 2}^{i_2 - 2r_2 = M} (-)^{i_2} \frac{f_{i_2} i_2!}{(4 \gamma_2)^{i_2}} \frac{(4 \gamma_2)^{r_2}}{r_2!} 
\end{align}
$

Due to the sums in the first and second line being restricted, they can not be directly translated to cross-correlations.
Instead, one can write

$
\begin{align}
v_{2r_1} &= \frac{1}{r_1! (4 \gamma_1)^{r_1}} \\
v_{2r_1 + 1} &= 0 \\
w_{i} &=  f_{i} i! (4 \gamma_1)^{i} 
\end{align}
$

To obtain

$a_L L! = \sum\limits_{j \leq i}^{i - j = L} w_{i} v_j = \sum\limits_{i, L \geq 0} w_{i} v_{i - L} \equiv \sum_i v_{L, i} w_i$

by promoting $v$ to a matrix. The rewriting for $b$ proceeds analogously. Then, defining 

$
\begin{align}
c_K &= \sum\limits_{L + M = K} a_L b_M = \text{Conv}[a, b]_K
\end{align}
$

we can write
$A_I = \sum\limits_{L + M - u = I} a_L b_M d_{I + u, u} = \sum\limits_{u \leq K}^{K - u = I} c_K d_{K, u} = \sum\limits_{I \geq 0}^{K} c_K d_{K, K - I} \equiv \sum\limits_{K} e_{I, K} c_K$
where 

$e_{I, K} &= \frac{ K! (-)^{K-I} p_x^{2I - K}}{(K-I)!(2I -K)!\delta^{I}}$
