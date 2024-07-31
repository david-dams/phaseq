import jax
import jax.numpy as jnp

def rho_closed_shell(vecs, N):
    return vecs[:, :N] @ vecs[:, :N].conj().T

def get_dc_mf(coulomb):
    def inner(rho):
        # 2 U_{abbd} <Ad>
        return 2 * jnp.diag( coulomb @ rho.diagonal() )
    return inner

def trafo_symmetric(overlap):
    overlap_vals, overlap_vecs = jnp.linalg.eigh(overlap)  
    sqrt_overlap = jnp.diag(overlap_vals**(-0.5))
    return overlap_vecs @ sqrt_overlap @ overlap_vecs.T

def scf_loop(overlap,
             kinetic,
             nuclear,
             f_trafo,
             f_rho,
             f_mean_field,
             mixing,
             limit,
             max_steps):
    """performs closed-shell scf calculation
    
    Args:
        overlap : NxN array
        nuclear : NxN array
        kinetic : NxN array
        f_trafo : Callable to produce the transformation matrix
        f_mean_field : Callable with signature `f : rho -> ham_int` to produce the mean field term
        f_rho : Callable with signature `f : evs -> ham` to produce the new density matrix
        mixing : float, percentage of old density to be mixed in the update
        precision : float, |rho - rho_old| < precision => break scf loop

    Returns: 
        rho : scf density matrix
    """

    def update(rho_old, step, error):
        """scf update"""

        # initial effective hamiltonian
        ham_eff = trafo.T @ (kinetic + nuclear + f_mean_field(rho_old)) @ trafo

        # diagonalize
        vals, vecs = jnp.linalg.eigh(ham_eff)    

        # build new density matrix
        rho = f_rho(trafo @ vecs)

        # update breaks
        error = jnp.linalg.norm(rho - rho_old)
        step = jax.lax.cond(error <= limit, lambda: step, lambda: step + 1, step)

        return rho, step, error
    
    def step(res):
        """single SCF update step"""
        return jax.lax.cond(res[-1] <= limit, lambda: res, update, res)

    # trafo orthogonalization
    trafo = f_trafo(overlap)

    # initial guess for the density matrix
    rho_old = jnp.zeros_like(ham)

    # scf loop
    rho, step, error = jax.lax.fori_loop(step, 0, max_steps, (rho_old, 0, jnp.inf))

    # intermediate stats
    print(f"After {steps} out of {max_steps}, scf finished with error {error}")
    
    return rho
