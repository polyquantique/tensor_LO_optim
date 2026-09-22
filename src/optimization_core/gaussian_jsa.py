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
    return jnp.exp(exponent)


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


# Callable scalar PDF function for gaussian JSA
def scaled_gaussian_jsa_vectorized_rsvd(r, mean, x_lim):
    """
    Returns a vectorized function that evaluates a multivariate Gaussian
    Joint Spectral Amplitude (JSA) on a grid.

    Parameters
    ----------
    r : float
        Correlation coefficient (off-diagonal value) for the covariance matrix.
        Must be between -1/(d-1) and 1 for invertibility.
    mean : array_like, shape (d,)
        Mean vector of the Gaussian.
        The dimensionality d is inferred from the length of this vector.

    Returns
    -------
    f : callable
        A function f(*coords) that evaluates the Gaussian JSA on a
        point or a grid. The arguments must be d arrays of identical shape (typically
        from jnp.meshgrid). The output is an array of the same shape
        containing the unnormalized JSA values.

    Notes
    -----
    - The returned JSA is NOT normalized. Apply continuous L^2 normalization
      externally using the correct volume element.
    - Scaling ensures that a fixed grid like [-4, 4]^d always captures the
      full Gaussian, even when the covariance is rotated or highly correlated.
    """
    d = mean.shape[0]
    cov = jnp.full((d, d), r)
    di = jnp.diag_indices(d)
    cov = cov.at[di].set(1.0)

    # Note: the rsvd library x_axis is defined between [0,1], so we need to re-center our multidimensional Gaussian

    # Calculate the gaussian largest width
    # eigvals, _ = jnp.linalg.eigh(cov)
    # sigma_max = jnp.sqrt(jnp.max(eigvals))
    # new_mean=(mean + 0.5)*sigma_max*(2*x_lim)

    def f(*coords):

        # Case 1: (N, d) point cloud
        if len(coords) == 1:
            pts = coords[0]
            pts_physical = (pts - 0.5) * (2 * x_lim)

            # Must be (N, d)
            if pts.ndim == 2 and pts.shape[1] == d:
                vals = multivariate_normal_pdf_func(pts_physical, mean, cov)
                return vals

            raise ValueError(
                f"Single argument must have shape (N,{d}), got {pts.shape}"
            )

        # Case 2: meshgrid arrays
        # ---------------------------------------------------------
        if len(coords) == d:
            # Stack into (..., d)
            # pts = jnp.stack(coords, axis=-1)
            coords_b = jnp.broadcast_arrays(*coords)
            pts = jnp.stack(coords_b, axis=-1)
            # Flatten to (N, d)
            pts_flat = pts.reshape(-1, d)
            pts_physical = (pts_flat - 0.5) * (2 * x_lim)

            # Evaluate PDF
            vals = multivariate_normal_pdf_func(pts_physical, mean, cov)

            # Reshape back to original grid shape

            return vals.reshape(coords_b[0].shape)

        raise ValueError(f"Expected {d} coordinates, got {len(coords)}")

    return f
