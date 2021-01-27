import jax.numpy as np
from jax.scipy.stats import multivariate_normal
from jax import random

def standard_normal(input_dim):
    """
    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.
    """
    mean = np.zeros(input_dim)
    covariance = np.eye(input_dim)

    def log_pdf(inputs):
        return multivariate_normal.logpdf(inputs, mean, covariance)

    def sample(key, num_samples=1):
        return random.multivariate_normal(key, mean, covariance, (num_samples,))

    return log_pdf, sample