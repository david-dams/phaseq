import dataclasses as dc
from collections import defaultdict
from functools import reduce

import jax
import jax.numpy as jnp

from phaseq import *

AtomChargeMap = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
        'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
        'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
        'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
    }

BasisMap ={
    "sto_3g" : {
        "pz" : { "coefficients" : jnp.array([ 0.155916, 0.607684, 0.391957 ]), 
                 "alphas" : jnp.array( [ 2.941249, 0.683483, 0.22229 ]),
                 "lmn" : jnp.array( [ 0,0,1 ]) } }
    }
    

class Basis:
    """holds gaussian basis expansions. 

    a basis is a dictionary mapping strings to JAX array tuples of the form 

    orbital_name : (coefficients, alphas, lmn)
    
    such that the i-th cgf is 

    [coefficients[i], position, lmn, alphas[i]]
    """
    def __init__(self, key):
        self.basis = BasisMap[key]
        
    @property
    def l_max(self):
        """maximum angular momentum in the basis set. Needed for JIT compilation of matrix elements"""
        return int(jnp.concatenate([c["lmn"] for c in self.basis.values()]).max()) + 1

@dc.dataclass
class Orbital:
    """represents an orbital"""
    name : str = None
    basis : jax.Array = dc.field(default_factory=lambda : jnp.array([0, 0, 0]))
    position : jax.Array = dc.field(default_factory=lambda : jnp.array([0, 0, 0]))

    @property
    def array(self):
        """converts orbital to cgf array representation of the form

        cgf = [ [coeff, primitive] ], where 
        
        coeff is the expansion coefficient for the primitive gaussian with array reprentation
        
        primitive = [pos, lmn, alpha]
        """

        orb = self.basis.basis[self.name]
        n_primitives = len(orb["alphas"])
        pos = jnp.array(self.position).astype(float)
        return jnp.stack( [jnp.concatenate([orb["coefficients"][i:(i+1)], pos, orb["lmn"], orb["alphas"][i:(i+1)]]) for i in range(n_primitives)] )
    
class Structure:
    """provides a convenience DSL for building a structure / molecule"""
    def __init__(self):
        self.orbitals = []
        self._nuclei_charges_positions = defaultdict(list)

    def __str__(self):
        return self.xyz
        
    def add_orbital(self, orb : Orbital, atom: str):
        self.orbitals.append(orb.array)
        charge = AtomChargeMap[atom]
        charge_pos = tuple([charge] + jnp.array(orb.position).astype(float).tolist())
        self._nuclei_charges_positions[atom].append(charge_pos)
        return self
        
    def add_orbitals(self, orbs : list[Orbital], atom : str):
        for orb in orbs:
            self.add_orbital(orb, atom)
        return self

    @property
    def xyz(self):
        total_atoms = len(self._nuclei_charges_positions)
        tail = '\n'.join(set([ ' '.join([name] + list(map(str,el[1:]))) for name, lst in self._nuclei_charges_positions.items() for el in lst]))        
        return f"{total_atoms}\n{tail}"

    @property
    def nuclei_charges_positions(self):
        """obtain nuclei charges and positions as an array of shape N x 4"""
        return jnp.array(reduce(lambda x,y : x+y, [list(set(values)) for values in self._nuclei_charges_positions.values()]))
        
    def scf(self, basis, **kwargs):
        
        sto3g = Basis("sto_3g")
        f_overlap, f_kinetic, f_nuclear, f_interaction = matrix_elements(sto3g.l_max)
        
        overlap_matrix = f_overlap(self.orbitals, self.orbitals)
        kinetic_matrix = f_kinetic(self.orbitals, self.orbitals)
        nuclear_matrix = f_nuclear(self.orbitals, self.orbitals, self.nuclei_charge_position)
        interaction_matrix = f_interaction(self.orbitals, self.orbitals, self.orbitals, self.orbitals)

        mf_default = lambda rho : jnp.einsum('abcd,cd->ab', interaction_matrix, rho)
        rho_default = lambda rho : rho_closed_shell(rho, overlap_matrix.shape[0] // 2)
        
        f_mean_field = kwargs.get("f_mean_field", mf_default)
        f_rho = kwargs.get("f_rho", rho_default)
        f_trafo = kwargs.get("f_trafo", trafo_symmetric)
        mixing = kwargs.get("mixing", 0.0)
        limit = kwargs.get("limit", 1e-8)
        max_steps = kwargs.get("max_steps", 100)

        return scf_loop(overlap,
                        kinetic,
                        nuclear,
                        f_trafo,
                        f_rho,
                        f_mean_field,
                        mixing,
                        limit,
                        max_steps)
