# Author: Gisell Osorio

import jax.numpy as jnp


def multivariate_normal_pdf_func(x, mean, cov):
    """Evaluate 3D multivariate normal PDF at points x (N,3)."""
    d = mean.shape[0]
    cov_inv = jnp.linalg.inv(cov)
    det_cov = jnp.linalg.det(cov)
    norm_const = 1.0 / jnp.sqrt((2 * jnp.pi) ** d * det_cov)
    diffs = x - mean
    exponent = -0.5 * jnp.sum(diffs @ cov_inv * diffs, axis=1)
    return norm_const * jnp.exp(exponent)


# Example: build a 3D grid and evaluate
def build_grid_and_pdf(n=50, mean=jnp.zeros(3), rho=0.5, x_lim=4):

    # Mean and covariance
    d = mean.shape[0]
    cov = jnp.full((d, d), rho)
    di = jnp.diag_indices(d)
    cov = cov.at[di].set(1.0)

    # cov = jnp.array([
    #     [1.0, rho, rho],
    #     [rho, 1.0, rho],
    #     [rho, rho, 1.0]
    # ])

    # Grid in [-x_lim,x_lim]^3
    lin = jnp.linspace(-x_lim, x_lim, n)
    X, Y, Z = jnp.meshgrid(lin, lin, lin, indexing="ij")
    points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Evaluate PDF
    pdf_vals = multivariate_normal_pdf_func(points, mean, cov)
    pdf_grid = pdf_vals.reshape((n, n, n))
    pdf_grid /= jnp.linalg.norm(pdf_grid)
    return X, Y, Z, pdf_grid


# Higher dimensions


def build_jsa_grid(n=50, mean=jnp.zeros(3), rho=0.5, x_lim=4):
    # Mean and covariance
    d = mean.shape[0]
    cov = jnp.full((d, d), rho)
    di = jnp.diag_indices(d)
    cov = cov.at[di].set(1.0)

    # Grid in [-x_lim,x_lim]^4
    axis = jnp.linspace(-x_lim, x_lim, n)
    # Create N‑dimensional meshgrid
    grid = jnp.meshgrid(*([axis] * d), indexing="ij")
    points = jnp.stack([g.ravel() for g in grid], axis=1)

    # Evaluate PDF
    pdf_vals = multivariate_normal_pdf_func(points, mean, cov)
    pdf_grid = pdf_vals.reshape(grid[0].shape)
    pdf_grid /= jnp.linalg.norm(pdf_grid)
    return points, pdf_grid
