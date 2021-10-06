from typing import List, Optional, Union

import arviz
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import linalg

"""
Code mainly contributed by Adrian Seyboldt (@aseyboldt) and Luciano Paz (@lucianopaz).
"""


def make_sum_zero_hh(N: int, a: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a householder transformation matrix that maps e_1 to a vector of all 1s.
    """
    if a is not None:
        if a.ndim != 1:
            raise NotImplementedError(
                "You idiot, that's not implemented yet, and it will never be!"
            )
        assert N == len(a), "The `a` vector must have same length as the GP cov matrix."
        a = a.astype(float)
    else:
        a = np.ones(N)
    
    e_1 = np.zeros(N)
    e_1[0] = 1
    a /= np.sqrt(a @ a)
    v = a + e_1
    v /= np.sqrt(v @ v)
    
    return np.eye(N) - 2 * np.outer(v, v)


def make_centered_gp_eigendecomp(
    time: np.ndarray,
    lengthscale: Union[float, str, List[Union[float, str]]],
    variance_limit: float = 0.95,
    variance_weight: Optional[List[float]] = None,
    kernel: str = "gaussian",
    metric: str = "euclidean",
    zerosum: bool = False,
    period: Optional[Union[float, str]] = None,
    a: Optional[np.ndarray] = None
):
    """
    Decompose the GP into eigen values and eigen vectors.

    Parameters
    ----------
    time : np.ndarray
        Array containing the time points of observations.
    lengthscale : float or str or list
        Length scale parameter of the GP. Set in the ``config`` dictionary.
        A list of lengthscales can be provided when using the Gaussian kernel.
        The corresponding covariance matrices will then be added to each other.
    variance_limit : float
        Controls how many of the eigen vectors of the GP are used. So, if
        ``variance_limit=1``, all eigen vectors are used.
    kernel : str
        Select the kernel function from the two available: gaussian or periodic.
    zerosum : bool
        Constrain all basis functions to sum(basis) = 0. The resulting GP will
        thus sum to 0 along the time axis.
    period : float or str
        Only used if the kernel is periodic. Determines the period of the kernel.
    a: Optional[np.ndarray] = None 
        NEED DOCSTING
    """

    ## Construct covariance matrix
    X = time[:, None]

    if kernel == "gaussian":
        if isinstance(lengthscale, (int, float, str)):
            lengthscale = [lengthscale]
            
        if variance_weight:
            assert len(variance_weight) == len(
                lengthscale
            ), "`variance_weight` must have the same length as `lengthscale`."
            variance_weight = np.asarray(variance_weight)
            assert np.isclose(variance_weight.sum(), 1.0), "`variance_weight` must sum to 1."
        else:
            variance_weight = np.ones_like(lengthscale)
        
        dists = []
        for ls in lengthscale:
            if isinstance(ls, str):
                ls = pd.to_timedelta(ls).to_timedelta64()
            dists.append(((X - X.T) / np.array(ls)) ** 2)
        
        cov = sum(
            w * np.exp(-dist / 2) for (w, dist) in zip(variance_weight, dists)
        ) / len(lengthscale)

    elif kernel == "periodic":
        if len(lengthscale) > 1:
            raise NotImplementedError(
                f"Multiple lengthscales can only be used with the Gaussian kernel."
            )
        elif variance_weight:
            raise NotImplementedError(
                f"`variance_weight` can only be used with the Gaussian kernel."
            )
        elif isinstance(period, str):
            period = pd.to_timedelta(period).to_timedelta64()
        
        dists = np.pi * ((time[:, None] - time[None, :]) / period)
        cov = np.exp(-2 * (np.sin(dists) / lengthscale) ** 2)

    else:
        raise ValueError(
            f"Unknown kernel = {kernel}. Accepted values are 'gaussian' and 'periodic'"
        )

    if zerosum:
        Q = make_sum_zero_hh(len(cov), a=a)
        D = np.eye(len(cov))
        D[0, 0] = 0

        # 1) Transform the covariance matrix so that the first entry
        # is the mean: A = Q @ cov @ Q.T
        # 2) Project onto the subspace without the mean: B = D @ A @ D
        # 3) Transform the result back to the original space: Q.T @ B @ Q
        cov = Q.T @ D @ Q @ cov @ Q.T @ D @ Q

    vals, vecs = linalg.eigh(cov)
    precision_limit_inds = np.logical_or(vals < 0, np.imag(vals) != 0)

    if np.any(precision_limit_inds):
        cutoff = np.where(precision_limit_inds[::-1])[0][0]
        vals = vals[len(vals) - cutoff :]
        vecs = vecs[:, vecs.shape[1] - cutoff :]

    if variance_limit == 1:
        n_eigs = len(vals)

    else:
        n_eigs = ((vals[::-1].cumsum() / vals.sum()) > variance_limit).nonzero()[0][0]

    return vecs[:, -n_eigs:] * np.sqrt(vals[-n_eigs:])


def make_gp_basis(time, gp_config, key=None, *, model=None):
    model = pm.modelcontext(model)

    if gp_config is None:
        gp_config = {
            "lengthscale": 8,
            "kernel": "gaussian",
            "zerosum": False,
            "variance_limit": 0.99,
        }
    else:
        gp_config = gp_config.copy()
    
    if (
        np.issubdtype(time.dtype, np.datetime64)
        or (str(time.dtype).startswith("datetime64"))
    ) and (
        gp_config["kernel"] == "gaussian"
        and "lengthscale" in gp_config
        and not isinstance(gp_config["lengthscale"], str)
    ):
        gp_config["lengthscale"] = f"{gp_config['lengthscale'] * 7}D"

    gp_basis_funcs = make_centered_gp_eigendecomp(time, **gp_config)
    n_basis = gp_basis_funcs.shape[1]
    dim = f"gp_{key}_basis"
    model.add_coords({dim: pd.RangeIndex(n_basis)})

    return gp_basis_funcs, dim


def sample_all(var_names: List[str], **sampler_kwargs):
    prior_checks = pm.sample_prior_predictive(var_names=var_names)
    posterior = pm.sample(return_inferencedata=False, **sampler_kwargs)
    post_checks = pm.sample_posterior_predictive(posterior, var_names=var_names)
    
    return arviz.from_pymc3(
        posterior, prior=prior_checks, posterior_predictive=post_checks
    )
