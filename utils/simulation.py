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
    num_shift_features = max(1, int(np.ceil(0.1 * dim)))
    for i in range(min(dim, num_shift_features)):
        mu_mx[i] = delta / (1 + 0.1 * i)

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


def generate_gaussian_mixture_q(dim, n, seed=None, mu3=None, diag=None):
    """
    Generate n samples from

        Q = 1/3 N(2*1, I_d)
          + 1/3 N(-2*1, I_d)
          + 1/3 N(mu3, Sigma3)

    where

        Sigma3 = I_d - D^2 + D E D

    with E the all-ones matrix and D = diag by default chosen so that
    Sigma3 = 0.8 I_d + 0.2 E, matching the notebook setting rho = 0.2.

    Parameters
    ----------
    dim : int
        Data dimension.
    n : int
        Number of samples.
    seed : int or None
        Random seed.
    mu3 : np.ndarray of shape (dim,) or None
        Mean of the third Gaussian component. If None, use the notebook's
        current default mu3 = 0.
    diag : np.ndarray of shape (dim, dim) or None
        Diagonal matrix D used to construct Sigma3. If None, use
        D = sqrt(0.2) I_d, matching the notebook's current rho = 0.2.

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

    if mu3 is None:
        mu3 = np.zeros(dim)
    else:
        mu3 = np.asarray(mu3, dtype=float)

    if diag is None:
        diag = np.sqrt(0.2) * I
    else:
        diag = np.asarray(diag, dtype=float)

    sigma12 = I
    sigma3 = I - diag @ diag + diag @ E @ diag

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


def generate_exponential(dim, n, beta=1.0, mu=0.0, seed=None):
    """
    Generate samples where each coordinate is i.i.d.

        X^i ~ Exp(beta) + mu

    Parameters
    ----------
    n : int
        Number of samples.
    dim : int, default=1
        Dimension of each sample.
    beta : float, default=1.0
        Exponential scale parameter.
    mu : float, default=0.0
        Additive location shift applied after sampling.
    seed : int or None, default=None
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n, dim), dtype float32
        Samples from the specified shifted exponential distribution.
    """
    rng = np.random.default_rng(seed)
    X = rng.exponential(scale=beta, size=(n, dim)) + mu
    return X.astype(np.float32)


def generate_gamma(shape, scale, dim, n, location_shift=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

    X = np.random.gamma(shape, scale, size=[n,dim])

    if location_shift is not None:
        X = X + location_shift

    return X.astype(np.float32)
