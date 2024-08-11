from collections import defaultdict
from functools import reduce
from itertools import product
import json

import jax
import jax.numpy as jnp

import phaseq as phaseq

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
    """Reads `json_file`, which must contain a basis set specification adhering to MolSSI BSE schema 0.1.
    Returns the corresponding internal basis representation of phaseQ.
    The internal basis representation is a dictionary.
    Atom names are keys, values are lists of arrays of dimension `number_of_gaussians' x 5.
    For example, `basis["atom"][i]` is the i-th orbital for "atom", defined as [coefficient, l, m, n, alpha].
    """

    def get_matching_angular_momentum(l):        
        """computes cartesian angular momentum arrays for l, i.e. l = 1 => [[0, 0, 1], [0, 1, 0], [1, 0, 0]]"""
        return list(map(list, filter(lambda x : sum(x) == l, product(range(l+1), range(l+1), range(l+1)))))

    def parse_cgf(lmn, exps, cs):
        """parses a single contracted gaussian function. Returns a N_gaussians x 5 Array"""
        return jnp.array(
            [ [float(c)] + lmn + [float(exps[i])] for i, c in enumerate(cs) ]
    )

    def parse_shell(orbital):
        """parses a single orbital shell. Returns an list of length N_cgf_in_shell, where each entry is a N_gaussians x 5 Array"""

        cs = orbital["coefficients"]
        am = orbital["angular_momentum"]
        return [parse_cgf(lmn, orbital["exponents"], cs[i]) for i, l in enumerate(am) for lmn in angular_momentum_map[l]]

    def parse_element(orbitals):
        """parses single element info. Returns a list of N_cgf, where each entry is a N_gaussians x 5 Array"""
        return reduce(lambda x,y : x + y, [parse_shell(o) for o in orbitals])

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

    # TODO: uff
    def matrix_functions(self):
        """Get functions to compute contracted gaussian matrixes. These functions functions have signature:

            f_one : (cgf1, cgf2) -> real
            f_two : (cgf1, cgf2) -> real
        
        Returns:
            f_overlap, f_kinetic, f_nuclear, f_interaction
        """

        l_max, n_orbs, nuclei_charges_positions = self.l_max, len(self.orbitals), self.nuclei_charges_positions
        
        # this is slightly awkward: a promoted nuclear element function maps orb x orb x nuc -> float, so we vmap and sum out the last axis, still closing over maximum angular momentum range
        f = jax.vmap( lambda x, n : phaseq.promote_nuclear(lambda a, b, c: phaseq.nuclear(a, b, c, 2*l_max))(x[0], x[1], n), (None, 0), 0 )
        func_nuclear = jax.jit(lambda x : f(x, nuclei_charges_positions).sum(axis = -1))
        func_kinetic = jax.jit(lambda x : phaseq.promote_one(lambda a, b: phaseq.kinetic(a, b, l_max + 1))(x[0], x[1]))
        func_overlap = jax.jit(lambda x : phaseq.promote_one(lambda a, b: phaseq.overlap(a, b, l_max))(x[0], x[1]))    
        func_interaction = jax.jit(lambda x : phaseq.promote_two(lambda a, b, c, d: phaseq.interaction(a, b, c, d, 2*l_max))(x[0], x[1], x[2], x[3]))

        func2 = lambda f : lambda lst: jnp.array(jax.tree.map(f, lst, is_leaf = lambda x: isinstance(x, tuple))).reshape(n_orbs, n_orbs)
        func4 = lambda f : lambda lst: jnp.array(jax.tree.map(f, lst, is_leaf = lambda x: isinstance(x, tuple))).reshape(n_orbs, n_orbs, n_orbs, n_orbs)

        return func2(func_overlap), func2(func_kinetic), func2(func_nuclear), func4(func_interaction)

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
        return int(jnp.max(jnp.concatenate([orb[:, 1:4] for orb in self.orbitals]))) + 1

    @property
    def one_electron_arguments(self):
        return list(product(self.orbitals,self.orbitals))
    
    @property
    def two_electron_arguments(self):
        return list(product(self.orbitals,self.orbitals, self.orbitals,self.orbitals))

    @property
    def n_electrons(self):
        return int(self.nuclei_charges_positions[:, 0].sum())

    @property
    def nuclear_repulsion_energy(self):
        r =  jnp.linalg.norm(self.nuclei_charges_positions[:, None, 1:] - self.nuclei_charges_positions[:, 1:], axis = -1)
        q =  self.nuclei_charges_positions[:, None, 0] * self.nuclei_charges_positions[:, 0]
        e = q / r
        return e[jnp.triu_indices_from(e, 1)].sum()
    
    def ground_state_energy(self, res : dict):
        """computes the ground state energy from a scf result dict"""
        return phaseq.energy( res["rho"], res["T"], res["V"], res["ham_eff"] ) + self.nuclear_repulsion_energy    
                
    def scf(self, mixing = 0.0, tolerance = 1e-8, steps = 100):
        """runs a self-consistent field calculation (restricted Hartree-Fock).

        Returns a dictionary containing the density matrix, density matrix, overlap, kinetic, nuclear, effective hamiltonian and interaction matrix"""
        
        overlap, kinetic, nuclear, interaction = self.matrix_functions()

        # currently, we do closed-shell only
        n_max = self.n_electrons // 2

        U = interaction(self.two_electron_arguments)
        
        res = phaseq.scf_loop(overlap(self.one_electron_arguments),
                               kinetic(self.one_electron_arguments),
                               nuclear(self.one_electron_arguments),
                               phaseq.trafo_canonical,
                               lambda v : phaseq.rho_closed_shell(v, n_max),
                               phaseq.get_mean_field_full(U),
                               mixing,
                               tolerance,
                               steps)

        return res | {"U" : U}
