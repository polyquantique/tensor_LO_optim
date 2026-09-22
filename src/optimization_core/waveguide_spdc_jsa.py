import jax.numpy as jnp


# Wavevector expansion: Phase mismatch calculation
def ldelk(ldelk0, tau1, taup, tauf, W):
    """
    w : array of shape (..., n)
        Frequencies w1, w2, ..., wn
    """
    S = jnp.sum(W, axis=-1)  # sum of all w_i
    Q = jnp.sum(W**2, axis=-1)  # sum of squares of w_i

    ldelk = ldelk0 + (tau1 / tauf) * (S) + ((taup**2) / (tauf**2)) * ((S) ** 2) + (Q)
    return ldelk


# JSA-Using the Taylor expansion of the wavevector
def scaled_spdc_jsa_vectorized(ldelk0, tau1, taup, tauf, T0=1 / jnp.sqrt(jnp.pi)):
    """
    Returns a callable f(*coords) that evaluates the SPDC JSA for an arbitrary number of frequency variables.
    Parameters
    ----------
    ldelk0, tau1, taup, tauf : floats
        Phase mismatch and walkoff/dispersion parameters.
    T0 : float
        Pump duration (used if pump envelop enabled).
    d : int
        Number of frequency variables (w1,...,wd).

    Returns
    -------
    f : callable
        f(x1, x2, ..., xd) where each xi is an array of identical shape.
        Returns an array of that same shape with the JSA values.
    """

    def f(*coords):
        # coords is a tuple: (w1, w2, ..., wd)
        W = jnp.stack(coords, axis=-1)  # shape (..., d)
        # Compute phase mismatch
        phase = ldelk(ldelk0, tau1, taup, tauf, W)

        # Core JSA
        # jsa=jnp.sinc((1/(jnp.pi))*(ldelk(tau1, taup,tauf,W)))*jnp.exp(-(1/2)*(T0**2/tauf**2)*jnp.sum(W,axis=-1)**2)

        jsa = jnp.sinc(phase / jnp.pi) * jnp.exp(
            -(1 / 2) * (T0**2 / tauf**2) * jnp.sum(W, axis=-1) ** 2
        )
        # jsa=jnp.sinc(phase/jnp.pi)
        return jsa

    return f


def scaled_spdc_jsa_chirped(ldelk0, tau1, taup, tauf, T0, C):
    """
    Returns a callable f(*coords) that evaluates the SPDC JSA for an arbitrary number of frequency variables.
    Parameters
    ----------
    ldelk0, tau1, taup, tauf : floats
        Phase mismatch and walkoff/dispersion parameters.
    T0 : float
        Pump duration (used if pump envelop enabled).
    C : float
        Chirp parameter.
    d : int
        Number of frequency variables (w1,...,wd).

    Returns
    -------
    f : callable
        f(x1, x2, ..., xd) where each xi is an array of identical shape.
        Returns an array of that same shape with the JSA values.
    """

    def f(*coords):
        # coords is a tuple: (w1, w2, ..., wd)
        W = jnp.stack(coords, axis=-1)  # shape (..., d)
        # Compute phase mismatch
        phase = ldelk(ldelk0, tau1, taup, tauf, W)
        pump = jnp.exp(
            -(T0**2 / ((2 * tauf**2) * (1 + 1j * C))) * jnp.sum(W, axis=-1) ** 2
        )

        jsa = jnp.sinc(phase / jnp.pi) * pump

        return jsa

    return f


# JSA-Using the Taylor expansion of the wavevector
def scaled_spdc_jsa_vectorized_rsvd(
    ldelk0, tau1, taup, tauf, domg, T0=1 / jnp.sqrt(jnp.pi)
):
    """
    Returns a callable f(*coords) that evaluates the SPDC JSA for an arbitrary number of frequency variables.
    Parameters
    ----------
    ldelk0, tau1, taup, tauf : floats
        Phase mismatch and walkoff/dispersion parameters.
    T0 : float
        Pump duration (used if pump envelop enabled).
    domg : float
        Limits for the frequency grid.
    d : int
        Number of frequency variables (w1,...,wd).

    Returns
    -------
    f : callable
        f(x1, x2, ..., xd) where each xi is an array of identical shape.
        Returns an array of that same shape with the JSA values.
    """

    def f(*coords):
        W_b = jnp.broadcast_arrays(*coords)
        W_s = jnp.stack(W_b, axis=-1)
        # W_flat = W_s.reshape(-1, d)
        W_scaled = (W_s - 0.5) * (2 * domg)

        # Compute phase mismatch
        phase = ldelk(ldelk0, tau1, taup, tauf, W_scaled)
        jsa = jnp.sinc(phase / jnp.pi) * jnp.exp(
            -(1 / 2) * (T0**2 / tauf**2) * jnp.sum(W_scaled, axis=-1) ** 2
        )
        return jsa.reshape(W_b[0].shape)

    return f
