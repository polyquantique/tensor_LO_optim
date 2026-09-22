# Author: Gisell Osorio

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxopt import ScipyMinimize
from jaxopt import ScipyBoundedMinimize

import scipy
from jax import jit
from jax import config

config.update("jax_debug_nans", True)
import numpy as np

from optimization_core import (
    objective_func_dim_penalty_param,
    objective_func_trust_constr,
    objective_penalty_norm_relaxation,
)


def optimizer_grad_desc_penalty(initial_seed, jsa, lambda_penalty, max_iters=1e4):
    """
    Parameters
    ----------
    initial_seed:  Initial seed-Local oscillator spectrum. 1d array containing complex numbers.
    jsa: JSA data
    lambda_penalty: Strength of the penalty term that enforces the normalization constraint on the local oscillator distribution.
    lr: Learning rate for the gradient descent optimization.
    max_iters: Maximum number of iterations for the optimization process.
    tol: Tolerance for convergence. The optimization will stop when the norm of the gradient is less than this value.

    Returns
    -------
    result.params: Optimized local oscillator spectrum. Half of the components correspond to the real part of the distribution, and the last half correspond to the imaginary part.
    result.state.fun_val: Value of the negative overlap between the jsa and the local oscillator distribution,              as defined in the paper.
    """

    f = initial_seed.copy()
    jsa = (
        jsa.copy()
    )  # Copy of jsa_data to avoid modifying the original due to numpy array mutability

    history = {"objective": []}

    def callback(params):
        # params is a JAX array → convert to numpy
        f_np = jnp.array(params)
        # recompute objective manually
        val = float(fun(f_np))
        history["objective"].append(val)

    def objective_func_dim_penalty_param_safe(f, jsa, lambda_penalty):

        val = objective_func_dim_penalty_param(f, jsa, lambda_penalty)
        val = jnp.where(jnp.isnan(val) | jnp.isinf(val), jnp.inf, val)
        return val

    # Objective and gradient
    fun = lambda f: objective_func_dim_penalty_param_safe(f, jsa, lambda_penalty)

    solver = ScipyMinimize(
        fun=fun,
        method="L-BFGS-B",
        maxiter=max_iters,
        tol=1e-8,
        jit=True,
        # callback=callback,
        options={"maxls": 100, "ftol": 1e-8, "gtol": 1e-8},
    )

    # Running the optimizer:
    result = solver.run(f)

    # Check success before returning
    if (
        not result.state.success
        or not np.isfinite(result.state.fun_val)
        or not np.isfinite(result.params).all()
    ):
        print(f"Converged: {result.state.success}")
        print(f"Warning: Optimization failed with status {result.state.status}")
        # Return a default value to handle the error here
        return None, None, None, False, result.state.status

    print(f"Converged: {result.state.success}")
    print(f"Exit status code: {result.state.status}")
    # # ans=(result.params,-1*result.state.fun_val,history)
    return (
        result.params,
        -1 * result.state.fun_val,
        history,
        result.state.success,
        result.state.status,
    )


# General L^p norm constraint builder
def p_norm_constraint(p):
    if p == 2:
        return {
            "type": "eq",
            "fun": lambda x: float(np.dot(x, x) - 1.0),
            "jac": lambda x: 2.0 * np.asarray(x, dtype=float),
        }
    else:
        # General L^p norm
        return {
            "type": "eq",
            "fun": lambda x: float(np.sum(np.abs(x) ** p) - 1.0),
            "jac": lambda x: p * np.abs(x) ** (p - 2) * np.asarray(x, dtype=float),
        }


def optimizer_basinhop_constrained(
    ini_seed, jsa, p_norm=2, n_iter=1000, temp=1.0, step_size=0.5, displ=False
):
    """
    Parameters
    ----------
    ini_seed: Initial seed-Local oscillator spectrum. 1d array containing complex numbers.
    jsa: JSA data
    p_norm: L^p norm to constrain the solution.
    n_iter: Number of iterations for the basinhopping optimization.
    temp: Temperature parameter for the basinhopping optimization, which controls the acceptance of worse solutions.
    step_size: Initial step size for the local optimization in the basinhopping algorithm.

    Returns
    -------
    Results of the basin hopping Scipy optimizer with 'L-BFGS-B' method for local
    optimization: (solution.x, solution.fun, solution.success*1.0) where
    solution.x is the local oscillator distribution that optimize the objective function,
    solution.fun is the negative JSA overlap value with an optimal local oscillator,
    solution.success*1.0 is 1.0 if the optimizer is successful and 0.0 otherwise
    """

    ini_seed = ini_seed.copy()
    jsa = jsa.copy()

    # ### The following function returns the function value and the gradient:
    # objective_func_value_grad = jit(jax.value_and_grad(objective_func_trust_constr, argnums=0))
    # objective_func_hess = jit(jax.hessian(objective_func_trust_constr, argnums=0))

    @jax.jit
    def objective_func_trust_constr_safe(f, jsa):
        # # optional: clip to avoid insane values
        # f = jnp.clip(f, -10.0, 10.0)

        val = objective_func_trust_constr(f, jsa)
        val = jnp.where(jnp.isnan(val) | jnp.isinf(val), jnp.inf, val)
        return val

    objective_func_value_grad = jax.jit(
        jax.value_and_grad(objective_func_trust_constr_safe, argnums=0)
    )
    objective_func_hess = jax.jit(
        jax.hessian(objective_func_trust_constr_safe, argnums=0)
    )

    def fun_np(x, jsa):
        val, grad = objective_func_value_grad(x, jsa)
        v = float(val)  # JAX scalar → Python float
        if not np.isfinite(v):
            print("[fun_np] Non-finite value:", v)
        return v

    def jac_np(x, jsa):
        val, grad = objective_func_value_grad(x, jsa)
        g = np.asarray(grad, dtype=float)  # JAX array → NumPy
        if not np.all(np.isfinite(g)):
            print("[jac_np] Non-finite gradient at x:", x)
        return g

    def hess_np(x, jsa):
        H = objective_func_hess(x, jsa)
        Hn = np.asarray(H, dtype=float)  # JAX array → NumPy
        if not np.all(np.isfinite(Hn)):
            print("[hess_np] Non-finite Hessian at x:", x)
        return Hn

    v0 = fun_np(ini_seed, jsa)
    g0 = jac_np(ini_seed, jsa)

    # print("fun(ini_seed) =", v0)
    # print("‖grad(ini_seed)‖ =", np.linalg.norm(g0))

    # Constraints and bounds for the optimization problem
    cons = p_norm_constraint(p_norm)
    dim = len(ini_seed)
    box_bounds = scipy.optimize.Bounds(
        [-5.0] * dim, [5.0] * dim
    )  # Box constraints to keep the solution in a reasonable range

    ### Options for the optimizer:
    minimizer_kwargs = {
        "method": "trust-constr",  # Local optimizer method that supports constraints
        "constraints": cons,
        "jac": jac_np,
        "hess": hess_np,  # TO DO: Removed the hessian only temporarily for the calculation of the 6D case
        "options": {
            "maxiter": 1000,
            "gtol": 1e-6,
            "xtol": 1e-6,
        },  # "barrier_tol":1e-7}, # TO DO: I lowered the tolerances for the 6D array only, update them to 1e-8 for the other cases and put , "barrier_tol":1e-7 only for the 6D case to loosen the constraint a little. maxiter original=1000
        "args": (jsa,),
        "bounds": box_bounds,
    }

    ### Optimizing using the basinhoping optimizer with the gradient and the initial
    #### seed passed to it (note that the objective function contains the normalization)
    solution = scipy.optimize.basinhopping(
        func=fun_np,
        x0=ini_seed,
        niter=n_iter,
        T=temp,
        stepsize=step_size,  # Initial guess
        target_accept_rate=0.5,  # adaptive tuning target
        minimizer_kwargs=minimizer_kwargs,
        disp=displ,  # Set to True to see detailed output from basinhopping
        niter_success=100,  # TO DO: Original value 200, reduced to 50 to accelarate a little the 6D calculation
    )

    mode_unnormalized = solution.x

    optimal_mode = mode_unnormalized

    local = solution.lowest_optimization_result

    print("Trust region method:")
    print("Final trust-constr success:", local.success)
    print("Exit status:", local.status)
    print("Message:", local.message)

    return (optimal_mode, -1 * solution.fun, solution, solution.success * 1.0)


# Homotopy continuation optimizers


def optimizer_grad_desc_penalty_norm_ramp(
    initial_seed, jsa, lambda_penalty, p, max_iters=1e4
):
    """
    Parameters
    ----------
    initial_seed:  Initial seed-Local oscillator spectrum. 1d array containing complex numbers.
    jsa: JSA data
    lambda_penalty: Strength of the penalty term that enforces the normalization constraint on the local oscillator distribution.
    p: The p value for the L^p norm constraint.
    lr: Learning rate for the gradient descent optimization.
    max_iters: Maximum number of iterations for the optimization process.
    tol: Tolerance for convergence. The optimization will stop when the norm of the gradient is less than this value.

    Returns
    -------
    result.params: Optimized local oscillator spectrum. Half of the components correspond to the real part of the distribution, and the last half correspond to the imaginary part.
    result.state.fun_val: Value of the negative overlap between the jsa and the local oscillator distribution,              as defined in the paper.
    """

    f = initial_seed.copy()
    jsa = (
        jsa.copy()
    )  # Copy of jsa_data to avoid modifying the original due to numpy array mutability

    history = {"objective": []}

    def callback(params):
        # params is a JAX array → convert to numpy
        f_np = jnp.array(params)
        # recompute objective manually
        val = float(fun(f_np))
        history["objective"].append(val)

    def objective_penalty_norm_relaxation_safe(f, jsa, lambda_penalty, p):

        val = objective_penalty_norm_relaxation(f, jsa, lambda_penalty, p)
        val = jnp.where(jnp.isnan(val) | jnp.isinf(val), jnp.inf, val)
        return val

    # Objective and gradient
    fun = lambda f: objective_penalty_norm_relaxation_safe(f, jsa, lambda_penalty, p)

    solver = ScipyMinimize(
        fun=fun,
        method="L-BFGS-B",
        maxiter=max_iters,
        tol=1e-8,
        jit=True,
        # callback=callback,
        options={"maxls": 100, "ftol": 1e-8, "gtol": 1e-8},
    )

    # Running the optimizer:
    result = solver.run(f)

    # Check success before returning
    if not result.state.success:
        print(f"Converged: {result.state.success}")
        print(f"Warning: Optimization failed with status {result.state.status}")
        # Return a default value to handle the error here
        return None, None, None, False, result.state.status

    print(f"Converged: {result.state.success}")
    print(f"Exit status code: {result.state.status}")
    # # ans=(result.params,-1*result.state.fun_val,history)
    return (
        result.params,
        -1 * result.state.fun_val,
        history,
        result.state.success,
        result.state.status,
    )


def optimizer_box_grad_desc_penalty_norm_ramp(
    initial_seed, jsa, lambda_penalty, p, max_iters=1e4
):
    """
    Parameters
    ----------
    initial_seed:  Initial seed-Local oscillator spectrum. 1d array containing complex numbers.
    jsa: JSA data
    lambda_penalty: Strength of the penalty term that enforces the normalization constraint on the local oscillator distribution.
    p: The p value for the L^p norm constraint.
    lr: Learning rate for the gradient descent optimization.
    max_iters: Maximum number of iterations for the optimization process.
    tol: Tolerance for convergence. The optimization will stop when the norm of the gradient is less than this value.

    Returns
    -------
    result.params: Optimized local oscillator spectrum. Half of the components correspond to the real part of the distribution, and the last half correspond to the imaginary part.
    result.state.fun_val: Value of the negative overlap between the jsa and the local oscillator distribution,              as defined in the paper.
    """

    f = initial_seed.copy()
    jsa = (
        jsa.copy()
    )  # Copy of jsa_data to avoid modifying the original due to numpy array mutability

    history = {"objective": []}

    def callback(params):
        # params is a JAX array → convert to numpy
        f_np = jnp.array(params)
        # recompute objective manually
        val = float(fun(f_np))
        history["objective"].append(val)

    def objective_penalty_norm_relaxation_safe(f, jsa, lambda_penalty, p):

        val = objective_penalty_norm_relaxation(f, jsa, lambda_penalty, p)
        val = jnp.where(jnp.isnan(val) | jnp.isinf(val), jnp.inf, val)
        return val

    # Objective and gradient
    fun = lambda f: objective_penalty_norm_relaxation_safe(f, jsa, lambda_penalty, p)

    # Box bounds to keep the solution in a reasonable range
    lower_bounds = jnp.full_like(f, -5.0)
    upper_bounds = jnp.full_like(f, 5.0)

    solver = ScipyBoundedMinimize(
        fun=fun,
        method="L-BFGS-B",
        maxiter=max_iters,
        tol=1e-8,
        jit=True,
        # callback=callback,
        options={"maxls": 100, "ftol": 1e-8, "gtol": 1e-8},
    )

    # Running the optimizer:
    result = solver.run(f, bounds=(lower_bounds, upper_bounds))

    # Check success before returning
    if not result.state.success:
        print(f"Converged: {result.state.success}")
        print(f"Warning: Optimization failed with status {result.state.status}")
        # Return a default value to handle the error here
        return None, None, None, False, result.state.status

    print(f"Converged: {result.state.success}")
    print(f"Exit status code: {result.state.status}")
    # # ans=(result.params,-1*result.state.fun_val,history)
    return (
        result.params,
        -1 * result.state.fun_val,
        history,
        result.state.success,
        result.state.status,
    )
