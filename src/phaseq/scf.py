import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import granad

# LDOS(r, w) ~ Tr[Im[G(r, r, w)]]
# (O + a) E = 1 => G = O (1 + O^{-1} a)^{-1}
# extended hubbard model $H = h_0 + U n_in_j + V n_in_j$
def energy(ham, pol, field, N):
    vals, vecs = jnp.linalg.eigh(ham + pol @ field)
    return vals[:N].sum()

def plot_force(flake):
    ham = flake.hamiltonian
    pol = flake.dipole_operator
    
    z = 0  # Position along the beam propagation direction
    w0 = 0.1  # Beam waist
    wavelength = 0.0001  # Wavelength of the laser

    e_of_x = lambda x : energy(ham, pol, gaussian_beam_profile(x, z, w0, wavelength), N)
    force = jax.grad(e_of_x)

    force[2].reshape()

    plt.matshow(force)
    plt.savefig("foo.pdf")
    
def dos(flake, scf_args, omegas):
    pol_bare = flake.get_polarizability_rpa(omegas, 0, hungry = 1)

    pol_cdw = flake.get_polarizability_rpa(omegas, 0, hungry = 1, args = scf_args)

    plt.plot(omegas, pol_cdw.imag / pol_cdw.imag.max(), label = 'cdw' )
    plt.plot(omegas, pol_bare.imag / pol_bare.imag.max()  , '--', label = 'fl')
    plt.xlabel(r'$\omega$')
    plt.ylabel("DOS")
    plt.legend()
    plt.savefig("cdw.pdf")
    

def scf_direct(ham, coulomb):
    """performs direct-channel only scf calculation
    
    Args:
         ham : NxN overlap matrix
         coulomb : NxN coulomb matrix

    Returns:
        h_eff : effective hamiltonian
    """

    # max number of iterations in sc loop
    max_steps = 100
    steps = 0
    
    # numerical precision of sc solution
    limit = 1e-8
        
    # initial guess for the density matrix
    rho_old = jnp.zeros_like(ham)

    # initial effective hamiltonian
    ham_eff = ham

    # diagonalize
    vals, vecs = jnp.linalg.eigh(ham_eff)    

    # build new density matrix
    N = int(ham.shape[0] / 2)
    rho = vecs[:, :N] @ vecs[:, :N].conj().T

    # iterate until convergence
    while jnp.linalg.norm(rho - rho_old) >= limit and max_steps > steps:

        # save last density matrix
        rho_old = rho

        # increment counter
        steps += 1

        # 2 U_{abbd} <Ad>
        direct_term = 2 * jnp.diag( coulomb @ rho.diagonal() )

        # new effective hamiltonian
        ham_eff = ham + direct_term

        # diagonalize
        vals, vecs = jnp.linalg.eigh(ham_eff)

        # new density matrix
        rho = vecs[:, :N] @ vecs[:, :N].conj().T

    print(f"After {steps} out of {max_steps}, scf finished with error {jnp.linalg.norm(rho - rho_old)}")
    
    return ham_eff, rho, vals, vecs

def get_metal(t, U, V):
    metal = (granad.Material("metal_1d")
             .lattice_constant(1.)
             .lattice_basis([
                 [1, 0, 0],
             ])
             .add_orbital_species("up", s = 1)
             .add_orbital(position=(0,), species="up")
             .add_orbital_species("down", s = -1)
             .add_orbital(position=(0,), species="down")
             .add_interaction(
                 "hamiltonian",
                 participants=("up", "up"),
                 parameters=[0.0, t],
             )
             .add_interaction(
                 "hamiltonian",
                 participants=("down", "down"),
                 parameters=[0.0, t],
             )
             .add_interaction(
                 "coulomb",
                 participants=("down", "up"),
                 parameters=[U, V],
             )
             .add_interaction(
                 "coulomb",
                 participants=("down", "up"),
                 parameters=[U, V],
             )
             .add_interaction(
                 "coulomb",
                 participants=("down", "down"),
                 parameters=[0., V],
             )
             .add_interaction(
                 "coulomb",
                 participants=("up", "up"),
                 parameters=[0., V],
             )
             )
    return metal

if __name__ == "__main__":

    # geometry
    cells = 60

    # parameters
    t = 1
    U = t / 2
    V = t 

    # set up material
    metal = get_metal(t, U, V)

    # set up chain
    flake = metal.cut_flake(cells)

    # half-filling
    flake.set_electrons(cells)    
    flake.set_open_shell()

    # mean-field
    ham_eff, rho, vals, vecs = scf_direct(flake.hamiltonian, flake.coulomb)

    scf_args = granad.TDArgs(
        hamiltonian = ham_eff,
        energies = vals,
        coulomb_scaled = flake.coulomb,
        initial_density_matrix = rho,
        stationary_density_matrix = flake.stationary_density_matrix,
        eigenvectors = vecs,
        dipole_operator = flake.dipole_operator,
        electrons = flake.electrons,
        relaxation_rate = 1/10,
        propagator = None,
        spin_degeneracy = 1.0,
        positions = flake.positions        
        )

    omegas = jnp.linspace(0, flake.energies.max()*2, 10)
    dos(flake, scf_args, omegas)
