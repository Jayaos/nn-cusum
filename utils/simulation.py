import numpy as np


def generate_gaussian_mean_shift_p(dim, n, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mu_mx = np.zeros(dim)
    sigma_mx = np.identity(dim)
    X = np.random.multivariate_normal(mu_mx, sigma_mx, n)

    return X.astype(np.float32)


def generate_gaussian_mean_shift_q(dim, n, delta, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mu_mx = np.zeros(dim)
    for i in range(min(dim, 3)):
        mu_mx[i] = delta / (i + 1)

    sigma_mx = np.identity(dim)
    X = np.random.multivariate_normal(mu_mx, sigma_mx, n)

    return X.astype(np.float32)


def generate_gaussian_mixture_p(dim, n, seed=None):
    """
    Generate n samples from

        P = 1/2 N(2*1, I_d) + 1/2 N(-2*1, I_d)

    Parameters
    ----------
    dim : int
        Data dimension.
    n : int
        Number of samples.
    seed : int or None
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n, dim), dtype float32
    """
    rng = np.random.default_rng(seed)

    one = np.ones(dim)
    mu1 = 2.0 * one
    mu2 = -2.0 * one
    sigma = np.eye(dim)

    z = rng.choice(2, size=n, p=[0.5, 0.5])

    n1 = np.sum(z == 0)
    n2 = np.sum(z == 1)

    X1 = rng.multivariate_normal(mean=mu1, cov=sigma, size=n1)
    X2 = rng.multivariate_normal(mean=mu2, cov=sigma, size=n2)

    X = np.empty((n, dim), dtype=np.float32)

    i1 = 0
    i2 = 0
    for i in range(n):
        if z[i] == 0:
            X[i] = X1[i1]
            i1 += 1
        else:
            X[i] = X2[i2]
            i2 += 1

    return X


def generate_gaussian_mixture_q(dim, n, seed=None):
    """
    Generate n samples from

        Q = 1/3 N(2*1, I_d)
          + 1/3 N(-2*1, I_d)
          + 1/3 N((13/5)*1, 0.8 I_d + 0.2 E)

    where E is the all-ones matrix.

    Parameters
    ----------
    dim : int
        Data dimension.
    n : int
        Number of samples.
    seed : int or None
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n, dim), dtype float32
    """
    rng = np.random.default_rng(seed)

    one = np.ones(dim)
    I = np.eye(dim)
    E = np.ones((dim, dim))

    mu1 = 2.0 * one
    mu2 = -2.0 * one
    mu3 = (13.0 / 5.0) * one

    sigma12 = I
    sigma3 = 0.8 * I + 0.2 * E

    z = rng.choice(3, size=n, p=[1/3, 1/3, 1/3])

    n1 = np.sum(z == 0)
    n2 = np.sum(z == 1)
    n3 = np.sum(z == 2)

    X1 = rng.multivariate_normal(mean=mu1, cov=sigma12, size=n1)
    X2 = rng.multivariate_normal(mean=mu2, cov=sigma12, size=n2)
    X3 = rng.multivariate_normal(mean=mu3, cov=sigma3, size=n3)

    X = np.empty((n, dim), dtype=np.float32)

    i1 = 0
    i2 = 0
    i3 = 0
    for i in range(n):
        if z[i] == 0:
            X[i] = X1[i1]
            i1 += 1
        elif z[i] == 1:
            X[i] = X2[i2]
            i2 += 1
        else:
            X[i] = X3[i3]
            i3 += 1

    return X


def generate_exponential_p(dim, n, seed=None):
    """
    Generate samples from p ~ f0, where each coordinate is i.i.d.

        f0^i ~ Exp(beta0),   beta0 = 1

    Parameters
    ----------
    n : int
        Number of samples.
    dim : int, default=1
        Dimension of each sample.
    seed : int or None, default=None
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n, dim), dtype float32
        Samples from p.
    """
    beta0 = 1.0
    rng = np.random.default_rng(seed)
    X = rng.exponential(scale=beta0, size=(n, dim))
    return X.astype(np.float32)


def generate_exponential_q(dim, n, seed=None):
    """
    Generate samples from q ~ f1, where each coordinate is i.i.d.

        f1^i ~ Exp(beta1) + mu_q,
        beta1 = 0.8,
        mu_q = beta0 - beta1 = 0.2

    Parameters
    ----------
    n : int
        Number of samples.
    dim : int, default=1
        Dimension of each sample.
    seed : int or None, default=None
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n, dim), dtype float32
        Samples from q.
    """
    beta0 = 1.0
    beta1 = 0.8
    mu_q = beta0 - beta1  # 0.2

    rng = np.random.default_rng(seed)
    X = rng.exponential(scale=beta1, size=(n, dim)) + mu_q
    return X.astype(np.float32)