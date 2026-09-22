import numpy as np
from scipy.optimize import minimize

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_device', jax.devices('cpu')[0])

def flatten_m2(t):
    n = t.ndim
    d = t.shape[0]
    T = t.reshape(d**(n-2), d**2)
    return T

def takagi_flattening(T, n, d):
    U, S, Vh = np.linalg.svd(T, full_matrices=False)
    K = len(S)

    u_tensors = []
    v_tensors = []

    for k in range(K):
        u = U[:, k].reshape((d,)*(n-2))
        v = Vh[k, :].reshape((d, d))
        u_tensors.append(u)
        v_tensors.append(v)

    return S, u_tensors, v_tensors

def v_caps_from_takagi(v_tensors):
    caps = []
    for v in v_tensors:
        # Takagi decomposition of symmetric matrix
        # Equivalent to SVD for symmetric complex matrices
        _, s, _ = np.linalg.svd(v)
        caps.append(s[0])
    return np.array(caps)

def matrix_caps(mats):
    caps = []
    for M in mats:
        svals = np.linalg.svd(M, compute_uv=False)
        caps.append(svals[0])   # largest singular value
    return np.array(caps)

def tensor_power_method(u, num_iters=200):
    d = u.shape[0]
    x = np.random.randn(d)
    x /= np.linalg.norm(x)

    n = u.ndim

    for _ in range(num_iters):
        # contraction over last n-1 indices
        y = u
        for _ in range(n-1):
            y = np.tensordot(y, x, axes=([1],[0]))
        x = y / np.linalg.norm(y)

    # final value
    val = u
    for _ in range(n):
        val = np.tensordot(val, x, axes=([0],[0]))
    return abs(val)

def compute_mu(u_tensors):
    return np.array([tensor_power_method(u) for u in u_tensors])

def knapsack_squared(weights, caps):
    idx = np.argsort(weights)[::-1]
    r = np.zeros_like(weights)
    budget = 1.0

    for i in idx:
        cap_sq = caps[i]**2
        if cap_sq <= budget:
            r[i] = caps[i]
            budget -= cap_sq
        else:
            r[i] = np.sqrt(budget)
            break

    return np.sum(weights * r**2), r


def knapsack_bound(sigma, mu, v_caps, tau):
    w_r = sigma * tau
    w_s = sigma / tau

    obj_r, _ = knapsack_squared(w_r, mu)
    obj_s, _ = knapsack_squared(w_s, v_caps)

    return 0.5 * (obj_r + obj_s)



def optimize_tau(sigma, mu, v_caps):
    K = len(sigma)
    tau0 = np.ones(K) # Initial guess for tau_k for each k

    def objective(tau):
        tau = np.abs(tau)
        return knapsack_bound(sigma, mu, v_caps, tau)

    res = minimize(objective, tau0, method='L-BFGS-B')
    tau_opt = np.abs(res.x)
    return tau_opt, res.fun

def tensor_knapsack_bound(t):
    n = t.ndim
    d = t.shape[0]

    # 1. flatten
    T = flatten_m2(t)

    # 2. Takagi/SVD
    sigma, u_tensors, v_tensors = takagi_flattening(T, n, d)

    # 3. caps for s_k
    v_caps = v_caps_from_takagi(v_tensors)

    # 4. recursive μ_k
    mu = compute_mu(u_tensors)

    # 5. optimize τ_k
    tau_opt, bound = optimize_tau(sigma, mu, v_caps)

    return bound, tau_opt, sigma, mu, v_caps


def tensor_knapsack_bound_4D(t):
    n = t.ndim
    d = t.shape[0]

    # 1. flatten
    T = flatten_m2(t)

    # 2. Takagi/SVD
    sigma, u_mats, v_mats = takagi_flattening(T, n, d)

    # 3. caps
    u_caps = matrix_caps(u_mats)
    v_caps = matrix_caps(v_mats)

    # 5. optimize τ_k
    tau_opt, bound = optimize_tau(sigma, u_caps, v_caps)

    return bound, tau_opt, sigma, u_caps, v_caps


# Knapsack bounds 5D

def flatten_2x3(t):
    d = t.shape[0]
    T = t.reshape(d**(2), d**3)
    return T

def initial_svd(T,d):
    U, S, Vh = jnp.linalg.svd(T, full_matrices=False)
    K = len(S)

    u_tensors = []
    v_tensors = []

    for k in range(K):
        u = U[:, k].reshape((d,)*2)
        v = Vh[k, :].reshape((d**2,d))
        u_tensors.append(u)
        v_tensors.append(v)

    return S, u_tensors, v_tensors

def u_caps_svd(u_tensors):
    caps = []
    for u in u_tensors:
        # svd decomposition of symmetric matrix
        s = jnp.linalg.svd(u, compute_uv=False)
        caps.append(s[0])
    return jnp.array(caps)

def v_caps_svd(v_tensors):
    caps = []
    for v in v_tensors:
        # svd decomposition of symmetric matrix
        s = jnp.linalg.svd(v, compute_uv=False)
        caps.append(s[0])
    return jnp.array(caps)

def tensor_knapsack_bound_5D(t):
    n = t.ndim
    d = t.shape[0]

    # 1. flatten 2x3
    T = flatten_2x3(t)

    # 2. Initial SVD
    sigma, u_mats, v_mats = initial_svd(T,d)

    # 3. caps
    u_caps = u_caps_svd(u_mats)
    v_caps = v_caps_svd(v_mats)

    # 5. optimize τ_k
    tau_opt, bound = optimize_tau(sigma, u_caps, v_caps)

    return bound, tau_opt, sigma, u_caps, v_caps

# Knapsack bounds 6D
def flatten_6d_first(t):
    d = t.shape[0]
    return t.reshape(d**4, d**2)

def svd_first(T1, d):
    U1, S1, Vh1 = np.linalg.svd(T1, full_matrices=False)
    u4_list = [U1[:, k].reshape(d, d, d, d) for k in range(len(S1))]
    v2_list = [Vh1[k, :].reshape(d, d) for k in range(len(S1))]
    return S1, u4_list, v2_list

def svd_second_per_k(u4_list, sigma1, d):
    sigma_pairs = []
    u2_list = []
    w2_list = []
    pair_k_index = []

    for k, u4 in enumerate(u4_list):
        T2 = u4.reshape(d**2, d**2)
        U2, A2, Vh2 = np.linalg.svd(T2, full_matrices=False)
        for j in range(len(A2)):
            alpha_kj = A2[j]
            u2 = U2[:, j].reshape(d, d)
            w2 = Vh2[j, :].reshape(d, d)
            sigma_pairs.append(sigma1[k] * alpha_kj)
            u2_list.append(u2)
            w2_list.append(w2)
            pair_k_index.append(k)

            print(f"Processing pair (k={k}, j={j})")

    return np.array(sigma_pairs), u2_list, w2_list, np.array(pair_k_index)

def knapsack_bound_6d(sigma_pairs, caps_u, caps_w,
                      sigma1, caps_v,
                      tau_u, tau_w, tau_v):
    w_u = sigma_pairs * tau_u
    w_w = sigma_pairs * tau_w
    w_v = sigma1 * tau_v

    obj_u, _ = knapsack_squared(w_u, caps_u)
    obj_w, _ = knapsack_squared(w_w, caps_w)
    obj_v, _ = knapsack_squared(w_v, caps_v)

    return (obj_u + obj_w + obj_v) / 3.0

def optimize_tau_6d(sigma_pairs, caps_u, caps_w,
                    sigma1, caps_v):
    P = len(sigma_pairs)
    K1 = len(sigma1)
    x0 = np.ones(2*P + K1)

    def objective(x):
        x = np.abs(x)
        tau_u = x[:P]
        tau_w = x[P:2*P]
        tau_v = x[2*P:]
        return knapsack_bound_6d(sigma_pairs, caps_u, caps_w,
                                 sigma1, caps_v,
                                 tau_u, tau_w, tau_v)

    res = minimize(objective, x0, method='L-BFGS-B')
    x_opt = np.abs(res.x)
    tau_u_opt = x_opt[:P]
    tau_w_opt = x_opt[P:2*P]
    tau_v_opt = x_opt[2*P:]
    return tau_u_opt, tau_w_opt, tau_v_opt, res.fun

def tensor_knapsack_bound_6d(t):
    d = t.shape[0]

    print("First flattening and SVD...")
    T1 = flatten_6d_first(t)
    sigma1, u4_list, v2_list = svd_first(T1, d)

    sigma_pairs, u2_list, w2_list, pair_k_index = svd_second_per_k(u4_list, sigma1, d)

    print("Computing caps for u...")
    caps_u = matrix_caps(u2_list)
    print("Computing caps for w...")
    caps_w = matrix_caps(w2_list)
    print("Computing caps for v...")
    caps_v = matrix_caps(v2_list)

    print("Optimizing tau values...")
    tau_u_opt, tau_w_opt, tau_v_opt, bound_min = optimize_tau_6d(
        sigma_pairs, caps_u, caps_w,
        sigma1, caps_v
    )

    return bound_min, tau_u_opt, tau_w_opt, tau_v_opt, sigma_pairs, caps_u, caps_w, sigma1, caps_v


