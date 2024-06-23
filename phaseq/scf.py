import jax
import jax.numpy as jnp
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

def ldos(args, omegas):
    pol = granad._numerics.rpa_polarizability_function(args, 0, 0, phi_ext = None)
    res = jax.lax.map(pol, omegas)    

    # compute greens tensor

    # compute ldos

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

    # numerical precision of sc solution
    limit = 1e-2
        
    # initial guess for the density matrix
    rho_old = jnp.zeros_like(ham)

    # initial effective hamiltonian
    ham_eff = ham

    # diagonalize
    vals, vecs = jnp.linalg.eigh(ham_eff)    

    # build new density matrix
    N = int(ham.shape[0] / 2)
    rho = vecs[:, :N].conj().T @ vecs[:, :N]

    # iterate until convergence
    while jnp.linalg.norm(rho - rho_old) >= limit and max_steps > steps:

        # save last density matrix
        rho_old = rho

        # increment counter
        steps += 1

        # mean field interaction picks out elements of coulomb matrix
        direct_term = coulomb

        # new effective hamiltonian
        ham_eff = ham + direct_term

        # diagonalize
        vals, vecs = jnp.linalg.eigh(ham_eff)

        # new density matrix
        rho = vecs[:, :N].conj().T @ vecs[:, :N]

    print(f"After {steps} out of {max_steps}, scf finished with error {jnp.linalg.norm(rho - rho_old)}")
    
    return ham_eff, rho

if __name__ == "__main__":
    flake = MaterialCatalog.get("metal_1d").cut_flake()

    ham_eff, rho = scf(flake.hamiltonian, flake.coulomb)

    # get normal ldos
    ldos_normal = ldos()

    # get cdw ldos
    ldos_cdw = ldos()

    plt.plot(omegas, ldos_normal)
    plt.plot(omegas, ldos_cdw)
    plt.savefig("ldos.pdf")
    
