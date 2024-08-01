from itertools import combinations_with_replacement, product

import jax
import jax.numpy as jnp

from pyqint import MoleculeBuilder,HF
from phaseq import *
        
# maps to expansion in terms of a list of cgfs like [c, lmn, alpha]
sto3g = {
    "C" : [
        # First S-orbital CGF
    jnp.array([
        [0.154329, 0, 0, 0, 71.616837],
        [0.535328, 0, 0, 0, 13.045096],
        [0.444635, 0, 0, 0, 3.530512]
    ]),
        # Second S-orbital CGF
    jnp.array([
        [-0.099967, 0, 0, 0, 2.941249],
        [0.399513, 0, 0, 0, 0.683483],
        [0.700115, 0, 0, 0, 0.22229]
    ]),
        # P_x orbital CGF
    jnp.array([
        [0.155916, 1, 0, 0, 2.941249],
        [0.607684, 1, 0, 0, 0.683483],
        [0.391957, 1, 0, 0, 0.22229]
    ]),
        # P_y orbital CGF
    jnp.array([
        [0.155916, 0, 1, 0, 2.941249],
        [0.607684, 0, 1, 0, 0.683483],
        [0.391957, 0, 1, 0, 0.22229]
    ]),
        # P_z orbital CGF
    jnp.array([
        [0.155916, 0, 0, 1, 2.941249],
        [0.607684, 0, 0, 1, 0.683483],
        [0.391957, 0, 0, 1, 0.22229]
    ])
    ],
    "H" :  [
        jnp.array([
            [0.154329, 0, 0, 0, 3.425251],
            [0.535328, 0, 0, 0, 0.623914],
            [0.444635, 0, 0, 0, 0.168855]
        ])
    ]
}

def get_mean_field(interaction_matrix):
    
    def inner(rho):
        return jnp.einsum('kl,ijlk->ij', rho, interaction_matrix) - 0.5 * jnp.einsum('kl,iklj->ij', rho, interaction_matrix)

    return inner
    

if __name__ == '__main__':
    mol = MoleculeBuilder().from_name('ch4')
    mol.name = 'CH4'
    ref = HF().rhf(mol, 'sto3g')
 
    ch4 = Structure.from_xyz(sto3g, "ch4.xyz", scale = 1.8897259886) # convert from angström to bohr

    # transform list of results into array
    to_matrix = lambda f, lst : jnp.array(jax.tree.map(f, lst, is_leaf = lambda x: isinstance(x, tuple))).reshape(9, 9)
    
    # matrix element functions
    l_max = ch4.l_max
    func_overlap = jax.jit(lambda x : promote_one(lambda a, b: overlap(a, b, l_max))(x[0], x[1]))
    func_kinetic = jax.jit(lambda x : promote_one(lambda a, b: kinetic(a, b, l_max+1))(x[0], x[1]))
    
    lst = list(product(ch4.orbitals, ch4.orbitals))
    overlap_matrix = to_matrix(func_overlap, lst)
    kinetic_matrix = to_matrix(func_kinetic, lst)

    # this is slightly awkward:
    # a promoted nuclear element function maps orb x orb x nuc -> float
    # so we vmap and sum out the last axis, still closing over maximum angular momentum range
    f = jax.vmap(
            lambda x, n : promote_nuclear(lambda a, b, c: nuclear(a, b, c, 2*l_max))(x[0], x[1], n), (None, 0), 0
        )
    func_nuclear = jax.jit(lambda x : f(x, ch4.nuclei_charges_positions).sum(axis = -1))
    nuclear_matrix = to_matrix(func_nuclear, lst)

    lst4 = list(product(ch4.orbitals, ch4.orbitals, ch4.orbitals, ch4.orbitals))
    func_interaction = jax.jit(lambda x : promote_two(lambda a, b, c, d: interaction(a, b, c, d, 2*l_max))(x[0], x[1], x[2], x[3]))
    interaction_matrix = jnp.array(jax.tree.map(func_interaction, lst4, is_leaf = lambda x: isinstance(x, tuple))).reshape(9, 9, 9, 9)

    rho = scf_loop(overlap_matrix,
                       kinetic_matrix,
                       nuclear_matrix,
                       trafo_symmetric,
                       lambda v : rho_closed_shell(v, 5),
                       get_mean_field(interaction_matrix),
                       0.,
                       1e-14,
                       100)

    assert jnp.abs(rho - ref["density"]).max() < 1e-8
