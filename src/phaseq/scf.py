import jax
import jax.numpy as jnp

def get_mean_field(interaction_matrix):
    
    def inner(rho):
        return jnp.einsum('kl,ijlk->ij', rho, interaction_matrix) - 0.5 * jnp.einsum('kl,iklj->ij', rho, interaction_matrix)

    return inner

def rho_closed_shell(vecs, N):
    return 2*vecs[:, :N] @ vecs[:, :N].T

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

    def energy(rho, ham_eff):
         return 0.5 * jnp.einsum('ij,ji', rho, kinetic + nuclear + ham_eff)
    
    def update(arg):
        """scf update"""
        
        rho_old, step, error = arg

        # initial effective hamiltonian
        ham_eff =  kinetic + nuclear + f_mean_field(rho_old)

        # diagonalize
        vals, vecs = jnp.linalg.eigh(trafo.T @ ham_eff @ trafo)    

        # build new density matrix
        rho = f_rho(trafo @ vecs) + mixing * rho_old

        # update breaks
        error = jnp.abs(energy(rho, ham_eff) - energy(rho_old, ham_eff))

        error = jnp.linalg.norm(rho - rho_old)

        step = jax.lax.cond(error <= limit, lambda x: step, lambda x: step + 1, step)

        return rho, step, error
    
    def step(idx, res):
        """single SCF update step"""
        return jax.lax.cond(res[-1] <= limit, lambda x: res, update, res)

    # trafo orthogonalization
    trafo = f_trafo(overlap)

    # initial guess for the density matrix
    rho_old = jnp.ones_like(overlap)

    # scf loop
    rho, steps, error = jax.lax.fori_loop(0, max_steps, step, (rho_old, 0, jnp.inf))

    # intermediate stats
    print(f"After {steps} out of {max_steps}, scf finished with error {error}")

    print(energy(rho, kinetic + nuclear + f_mean_field(rho)))
    
    return rho
