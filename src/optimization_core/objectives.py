# Author: Gisell Osorio

import jax
import jax.numpy as jnp


def contract_nd(jsa, fcomp):
    N = jsa.ndim
    # Build index string: 'abcd' for 4D, 'abcde' for 5D, etc.
    idx = "".join(chr(ord("a") + i) for i in range(N))
    # Build einsum pattern: 'abcd,a,b,c,d'
    pattern = f"{idx}," + ",".join(idx[i] for i in range(N))
    overlap = jnp.einsum(pattern, jsa, *([jnp.conj(fcomp)] * N))
    # Check if result is finite
    is_bad = ~jnp.isfinite(overlap)
    # Make sure both branches return the same JAX array type
    nan_val = jnp.array(jnp.nan, dtype=overlap.dtype)
    # If bad → return nan; else → return overlap
    return jax.lax.cond(is_bad, lambda _: nan_val, lambda _: overlap, operand=None)


def objective_func_dim_penalty_param(
    f, jsa, lambda_penalty=1e6
):  # lambda_penalty = 1e6 is a default value, making lambda_penalty an optional argument.
    """
    Parameters
    ----------
    f:  Local oscillator spectrum. 1d array containing complex numbers,
        half of the components correspond to the real part of the distribution,
        and the last half correspond to the imaginary part.

    jsa: JSA data

    Returns
    -------
    Negative overlap between the jsa and the local oscillator function,
    as defined in the paper.
    """
    f = (
        f.copy()
    )  # Copy of f to avoid modifying the original due to numpy array mutability
    jsa = (
        jsa.copy()
    )  # Copy of jsa to avoid modifying the original due to numpy array mutability

    fre, fim = jnp.split(f, [int(f.size / 2)])
    fcomp = fre + fim * 1j

    # Strength of the penalty
    fnorm = jnp.linalg.norm(fcomp) ** 2
    penalty = lambda_penalty * (fnorm - 1) ** 2

    jsa_overlap = contract_nd(jsa, jnp.conj(fcomp))

    # We maximize (overlap - lambda * ||fcom||^2)  →  minimize negative
    return -(
        jnp.real(jsa_overlap) - penalty
    )  # We define it negative, because we are using the minimize solver, but our goal is to actually maximize it


# Objective function
def objective_func_trust_constr(f, jsa):
    """
    Parameters
    ----------
    f:  Local oscillator spectrum. 1d array containing complex numbers,
        half of the components correspond to the real part of the distribution,
        and the last half correspond to the imaginary part.

    jsa: JSA data
    p_norm: The p value for the L^p norm constraint. Default is 2, which corresponds to the standard norm used in quantum mechanics.

    Returns
    -------
    Negative overlap between the jsa and the local oscillator function,
    as defined in the paper.
    """
    f = f.copy()
    jsa = jsa.copy()

    fre, fim = jnp.split(f, [int(f.size / 2)])
    fcomp = fre + fim * 1j

    # jsa_overlap=jnp.abs(jnp.einsum("ijk,i,j,k",jsa,fcomp,fcomp,fcomp))
    jsa_overlap = jsa
    for axis in range(jsa.ndim):
        jsa_overlap = jnp.tensordot(jsa_overlap, jnp.conj(fcomp), axes=([0], [0]))

    return -jnp.real(
        jsa_overlap
    )  # We define it negative, because we are using the minimize solver, but our goal is to actually maximize it


def objective_penalty_norm_relaxation(f, jsa, lambda_penalty=1e6, p=2):
    f = f.copy()
    jsa = jsa.copy()

    fre, fim = jnp.split(f, [f.size // 2])
    fcomp = fre + 1j * fim

    # p-norm relaxation
    fnorm = jnp.sum(jnp.abs(fcomp) ** p) ** (1.0 / p)
    penalty = (
        lambda_penalty * (fnorm - 1.0) ** 2
    )  # Squaring the constraint makes it smoothly differentiable

    # Tensor contraction
    jsa_overlap = jsa
    for axis in range(jsa.ndim):
        jsa_overlap = jnp.tensordot(jsa_overlap, jnp.conj(fcomp), axes=([0], [0]))

    return -(jnp.real(jsa_overlap) - penalty)
