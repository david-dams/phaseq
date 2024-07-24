import jax
import jax.numpy as jnp

from phaseq import *

# convention used for built-in gaussian basis: tuples have the form (coefficients, alphas, lmn), where lmn is exponent of cartesian coordinates x,y,z
sto_3g = {
    "pz" : (jnp.array([ 0.155916, 0.607684, 0.391957 ]), 
            jnp.array( [ 2.941249, 0.683483, 0.22229 ]),
            jnp.array( [ 0,0,1 ]) )
    }

def promote_one(f):
    """one electron matrix element promotion"""
    def element(coeff1, coeff2, g1, g2):        
        c1 = get_norms_coefficients(g1, coeff1)
        c2 = get_norms_coefficients(g2, coeff2)        
        return jnp.einsum('k,l,kl->', c1, c2, f_vmapped(g2, g1))
    
    f_vmapped = jax.vmap(jax.vmap(lambda g1, g2 : f(g1, g2), (0, None), 0), (None, 0), 0)
    
    return element

def promote_two(f):
    """two electron matrix element promotion"""
    def element(coeff1, coeff2, coeff3, coeff4, g1, g2, g3, g4):
        c1 = get_norms_coefficients(g1, coeff1)
        c2 = get_norms_coefficients(g2, coeff2)
        c3 = get_norms_coefficients(g3, coeff3)
        c4 = get_norms_coefficients(g4, coeff4)
        
        return jnp.einsum('i,j,k,l,ijkl->', c1, c2, c3, c4, f_vmapped(g1, g2, g3, g4))
    
    f_vmapped = jax.vmap(jax.vmap(lambda g1, g2, g3, g4 : f(g1, g2, g3, g4), (0, None), 0), (None, 0), 0)
    
    return element

def get_norms_coefficients(gaussians, expansion):
    """Computes the exact coefficients needed for converting primtive to contracted gaussian matrix elements. These simply include the norms of the primitive Gaussians.
    
    Args: 

       gaussians : N x 7 array of primitive gaussians
       expansion : N - array of basis coefficients     

    Returns:
    
       N - array of norms * expansion
    """
    return expansion * jax.vmap(norm, (0,))(gaussians)
        

def get_atomic_charge(atom):
    charges = {
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
    return charges[atom]

def expand_gaussian(orb_list, expansion):
    """converts OrbitalList into a representation compatible with gaussian integration.
    
    Args:
    orb_list: OrbtalList 
    expansion: dictionary mapping group ids to arrays
    
    Returns:
    """
    
    # translate orbital list to array representation
    orbitals = jnp.array( [jnp.concatenate([jnp.concatenate(expansion[o.group_id]), o.position]) for o in orb_list] )

    # size of coefficients for unpacking array arguments
    coeff_size = expansion[orb_list[0].group_id][0].size

    # array representation of nuclei, index map to keep track of orbitals => nuclei
    nuclei_positions, idxs, orbitals_to_nuclei = jnp.unique(orbitals[:,:3], axis = 0, return_index = True, return_inverse = True)
    nuclei_charges = jnp.array([float(get_atomic_charge(o.atom_name)) for o in orb_list])[idxs]
    nuclei = jnp.hstack( [nuclei_positions, nuclei_charges[:,None]] )

    # max l for static loops compatible with JIT+reverse AD
    l_max = orbitals[3:6].max()

    # gto funcs 
    overlap, kinetic, nuclear, repulsion = get_gaussian_functions(l_max)

    # cgf expansion
    overlap = one_body_wrapper(get_overlap(l_max), coeff_size)
    kinetic = one_body_wrapper(get_kinetic(l_max), coeff_size)
    nuclear = one_body_wrapper(get_nuclear(l_max), coeff_size)
    repulsion = two_body_wrapper(get_repulsion(l_max), coeff_size)
    
    return overlap, kinetic, nuclear, repulsion, orbitals, nuclei, orbitals_to_nuclei
