import jax.numpy as jnp

def rbf_kernel(X, sigma=1.0):
    # Compute the pairwise squared Euclidean distances
    pairwise_dists = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    # Apply the RBF kernel function
    values = jnp.exp(-pairwise_dists / (2 * sigma ** 2))
    return values

def compute_HSIC(Z_b, Z_c):
    n = Z_b.shape[0]
    # Compute kernel matrices
    K = rbf_kernel(Z_b)
    L = rbf_kernel(Z_c)
    # Implement the HSIC formula
    term1 = (1 / (n ** 2)) * jnp.sum(K * L)
    term2 = (1 / (n ** 4)) * jnp.sum(K) * jnp.sum(L)
    term3 = (2 / (n ** 3)) * jnp.sum(K @ L)
    HSIC_n = term1 + term2 - term3
    return HSIC_n * n