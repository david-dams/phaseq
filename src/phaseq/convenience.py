from collections import defaultdict
from functools import reduce
from itertools import product
import json

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

def basis_from_json(json_file):
    """converts json format basis set adhering to molssi_bse_schema, schema_version 0.1 to phaseq's internal basis representation"""

    def get_matching_angular_momentum(l):        
        """computes cartesian angular momentum arrays for l, i.e. l = 1 => [[0,0,1], [0, 1, 0], [1 0, 0]]"""
        return list(map(list, filter(lambda x : sum(x) == l, product(range(l+1), range(l+1), range(l+1)))))

    def parse_cgf(l, exps, cs):
        return [[float(c)] + am + [float(exps[i])] for i, c in enumerate(cs) for am in angular_momentum_map[l]]

    def parse_orbital(orbital):
        """parses a single orbital. Returns a N_gaussians x 8 Array"""

        cs = orbital["coefficients"]
        am = orbital["angular_momentum"]
        return jnp.array(reduce(lambda x,y : x+y,
                      [parse_cgf(l, orbital["exponents"], cs[i]) for i, l in enumerate(am)]
                      ))

    def parse_element(orbitals):
        """parses single element info. Returns N_orbitals x N_gaussians x 8 Array."""

        return [parse_orbital(o) for o in orbitals]

    def get_l_max(basis_json):
        return max(reduce(lambda x,y: x+y, [z["angular_momentum"] for y in [x["electron_shells"] for x in basis_json["elements"].values()] for z in y]))
    
    with open(json_file, "r") as f:
        basis_json = json.load(f)

    angular_momentum_map = [get_matching_angular_momentum(l) for l in range(get_l_max(basis_json)+1)]
    atom_charge_list = list(AtomChargeMap.keys())

    basis = {}
    for number, info in basis_json["elements"].items():
        basis[atom_charge_list[int(number)-1]] = parse_element(info["electron_shells"])

    return basis

class Structure:
    """provides a convenience DSL for building a structure / molecule"""
    def __init__(self):
        self.orbitals = []
        self._nuclei_charges_positions = defaultdict(list)

    def __str__(self):
        return self.xyz
        
    def add_orbital(self, orb : jax.Array, position : list, atom : str):                
        position = jnp.array(position).astype(float)

        assert position.shape == (3,), "Position must be 3-dim vector"
        assert orb.ndim == 2 and orb.shape[-1] == 5, "Orb must be N x 5 array"
        
        orb = jnp.insert(orb, jnp.array([1]), position, axis=1)
        self.orbitals.append(orb)
        
        charge = AtomChargeMap[atom]
        charge_pos = tuple([charge] + position.tolist())
        self._nuclei_charges_positions[atom].append(charge_pos)
        
        return self
        
    def add_orbitals(self, orbs : jax.Array, position : list, atom : str):
        for orb in orbs:
            self.add_orbital(orb, position, atom)
        return self

    def add_atom(self, atom, position, name):
        self.add_orbitals(atom, position, name)
        return self

    def process_line(self, basis : dict, line : str, fac : float):
        splt = line.split()
        if len(splt) != 4:
            return
        atom, position = splt[0], jnp.array(list(map(lambda x : fac * float(x), splt[1:])))
        self.add_atom(basis[atom], position, atom)

    @classmethod
    def from_xyz(cls, basis : dict, filename : str, scale = 1.0):
        c = cls()
        with open(filename, "r") as f:
            for l in f.readlines():
                c.process_line(basis, l, scale)
        return c

    @property
    def xyz(self):
        tail = set([ ' '.join([name] + list(map(str,el[1:]))) for name, lst in self._nuclei_charges_positions.items() for el in lst])
        total_atoms = len(tail)
        tail = '\n'.join(tail)
        return f"{total_atoms}\n{tail}"

    @property
    def nuclei_charges_positions(self):
        """obtain nuclei charges and positions as an array of shape N x 4"""
        return jnp.array(reduce(lambda x,y : x+y, [list(set(values)) for values in self._nuclei_charges_positions.values()]))

    @property
    def l_max(self):
        """maximum angular momentum used by the orbitals. Needed for JIT compilation of matrix elements"""
        return int(jnp.max(jnp.concatenate([ orb[:, 1:4] for orb in self.orbitals]))) + 1
        
    def scf(self, **kwargs):
        
        f_overlap, f_kinetic, f_nuclear, f_interaction = matrix_elements_vmapped(self.l_max)
        
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
