""" Utility functions for helping compute GLMNET models
"""
import numpy as np


def mse_path(X, y, coefs, intercepts):
    """ Return mean squared error for sets of estimated coefficients

    Args:
        X (np.ndarray): 2D (n_obs x n_features) design matrix
        y (np.ndarray): 1D dependent variable
        coefs (np.ndarray): 1 or 2D array of coefficients estimated from
            GLMNET using one or more ``lambdas`` (n_coef x n_lambdas)
        intercepts (np.ndarray): 1 or 2D array of intercepts from
            GLMNET using one or more ``lambdas`` (n_lambdas)

    Returns:
        np.ndarray: mean squared error as 1D array (n_lambdas)

    """
    coefs = np.atleast_2d(coefs)
    intercepts = np.atleast_1d(intercepts)

    resid = y[:, np.newaxis] - (np.dot(X, coefs) + intercepts)
    mse = (resid ** 2).mean(axis=0)

    return mse


def IC_path(X, y, coefs, intercepts, criterion='AIC'):
    """ Return AIC, BIC, or AICc for sets of estimated coefficients

    Args:
        X (np.ndarray): 2D (n_obs x n_features) design matrix
        y (np.ndarray): 1D dependent variable
        coefs (np.ndarray): 1 or 2D array of coefficients estimated from
            GLMNET using one or more ``lambdas`` (n_coef x n_lambdas)
        intercepts (np.ndarray): 1 or 2D array of intercepts from
            GLMNET using one or more ``lambdas`` (n_lambdas)
        criterion (str): AIC (Akaike Information Criterion), BIC (Bayesian
            Information Criterion), or AICc (Akaike Information Criterion
            corrected for finite sample sizes)

    Returns:
        np.ndarray: information criterion as 1D array (n_lambdas)

    Note: AIC and BIC calculations taken from scikit-learn's LarsCV

    """
    coefs = np.atleast_2d(coefs)

    n_samples = y.shape[0]

    criterion = criterion.lower()
    if criterion == 'aic' or criterion == 'aicc':
        K = 2
    elif criterion == 'bic':
        K = np.log(n_samples)
    else:
        raise ValueError('Criterion must be either AIC, BIC, or AICc')

    mse = mse_path(X, y, coefs, intercepts)

    df = np.zeros(coefs.shape[1], dtype=np.int)
    for k, coef in enumerate(coefs.T):
        mask = np.abs(coef) > np.finfo(coef.dtype).eps
        if not np.any(mask):
            continue
        df[k] = np.sum(mask)

    with np.errstate(divide='ignore'):
        criterion_ = n_samples * np.log(mse) + K * df
        if criterion == 'aicc':
            criterion_ += (2 * df * (df + 1)) / (n_samples - df - 1)

    return criterion_
