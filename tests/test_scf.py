from pyqint import MoleculeBuilder,HF
from phaseq import *
import dataclasses as dc

import jax
import jax.numpy as jnp

if __name__ == '__main__':
    sto3g = Basis("sto_3g")
    f_overlap, f_kinetic, f_nuclear, f_interaction = matrix_elements(sto3g.l_max)

    hs = [Orbital(name = "pz", basis = sto3g, position = pos ) for pos in [[0, 0, 0]] ]
    c = Orbital(name = "pz", basis = sto3g, position = [0., 0, 0] )
    ch4 = Structure().add_orbitals(hs, "H").add_orbital(c, "C")
    import pdb; pdb.set_trace()


    overlap_matrix = f_overlap(ch4.orbitals, ch4.orbitals)
    kinetic_matrix = f_kinetic(ch4.orbitals, ch4.orbitals)
    nuclear_matrix = f_nuclear(ch4.orbitals, ch4.orbitals, ch4.nuclei_charge_position)
    interaction_matrix = f_interaction(ch4.orbitals, ch4.orbitals, ch4.orbitals, ch4.orbitals)

    mf = lambda rho : jnp.einsum('abcd,cd->ab', interaction_matrix, rho)

    rho = scf_loop(overlap_matrix,
                   kinetic_matrix,
                   nuclear_matrix,
                   lambda v : trafo_symmetric(v, overlap_matrix.shape[0] // 2),
                   rho_closed_shell,
                   mf)
    
    
    mol = MoleculeBuilder().from_name('ch4')
    mol.name = 'CH4'
    res = HF().rhf(mol, 'sto3g')
    print()
    print('Kinetic energy: ', res['ekin'])
    print('Nuclear attraction energy: ', res['enuc'])
    print('Electron-electron repulsion: ', res['erep'])
    print('Exchange energy: ', res['ex'])
    print('Repulsion between nuclei: ', res['enucrep'])
    print()
    print('Total energy: ', res['energy'])
    print('Sum of the individual terms: ',
          res['ekin'] + res['enuc'] + res['erep'] + res['ex'] + res['enucrep'])
