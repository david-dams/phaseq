from phaseq import *
from pyqint import *

if __name__ == '__main__':
    # primitive gaussian is always [pos, lmn, alpha]
    # gaussian1 = jnp.array( [-0.1, 0.3, 0.7, 2, 1, 3, 0.2] )
    # gaussian2 = jnp.array( [0.1, 0.4, 0.1, 2, 0, 5, 0.1] )

    # gaussian1 = jnp.array( [-0.1, 0.3, 0.7, 1, 1, 1, 0.2] )
    # gaussian2 = jnp.array( [0.1, 0.4, 0.1, 1, 0, 1, 0.1] )

    gaussian1 = jnp.array( [0.2, 0.5, 10, 1, 0, 0, 0.1] )
    gaussian2 = jnp.array( [0.1, 0.3, 0.7, 1, 0, 0, 0.1] )

    # a = overlap(l_arr, gaussian1, gaussian2, jnp.arange(2*l_arr.max()+1))

    positions = jnp.array([ [x, y, 0] for x in range(4) for y in range(4) ]) + 0.1

    # Define a function that maps over the first axis
    vmap_first = jax.vmap(nuclear_matrix, in_axes=(0, None, None))

    # Define a function that maps over the second axis, now vmap_first will be vectorized over the second argument
    vmap_second = jax.vmap(vmap_first, in_axes=(None, 0, None))

    # Define a function that maps over the third axis, now vmap_second will be vectorized over the third argument
    vmap_third = jax.vmap(vmap_second, in_axes=(None, None, 0))

    # Now apply vmap_third to the positions array
    a = vmap_third(positions, positions, positions)

    import matplotlib.pyplot as plt
    plt.matshow(jnp.sum(a, axis=2))
    plt.xlabel("# orbital")
    plt.ylabel("# orbital")
    plt.title(r"square lattice classical $H_{\text{e-ph}}, \delta R$ = const.")
    plt.savefig("phonon.pdf")
    
    
    # fun = lambda x : jnp.sum(vmap_third(positions, positions, x), axis = 2)
    # grad = jax.jacrev(fun)
    # g = grad(positions)
    # import pdb; pdb.set_trace()

