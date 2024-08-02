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

if __name__ == '__main__':
    mol = MoleculeBuilder().from_name('ch4')
    mol.name = 'CH4'
    ref = HF().rhf(mol, 'sto3g')
 
    ch4 = Structure.from_xyz(
        sto3g,
        "ch4.xyz",
        scale = 1.8897259886  # convert from angström to bohr
    )

    rho = ch4.scf(tolerance = 1e-14)
    
    assert jnp.abs(rho - ref["density"]).max() < 1e-8
