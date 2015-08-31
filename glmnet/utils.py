""" Utility functions for helping compute GLMNET models
"""
import numpy as np


def mse_path(coefs, X, y):
    """ Return mean squared error for sets of estimated coefficients

    Args:
        coefs (np.ndarray): 1 or 2D array of coefficients estimated from
            GLMNET using one or more ``lambdas`` (n_coef x n_lambdas)
        X (np.ndarray): 2D (n_obs x n_features) design matrix
        y (np.ndarray): 1D dependent variable

    Returns:
        np.ndarray: mean squared error as 1D array (n_lambdas)

    """
    if coefs.ndim == 1:
        coefs = coefs[:, np.newaxis]

    resid = y[:, np.newaxis] - np.dot(X, coefs)
    mse = (resid ** 2).mean(axis=0)

    return mse


def IC_path(coefs, X, y, criterion='AIC'):
    """ Return AIC, BIC, or AICc for sets of estimated coefficients

    Args:
        coefs (np.ndarray): 1 or 2D array of coefficients estimated from
            GLMNET using one or more ``lambdas`` (n_coef x n_lambdas)
        X (np.ndarray): 2D (n_obs x n_features) design matrix
        y (np.ndarray): 1D dependent variable
        criterion (str): AIC (Akaike Information Criterion), BIC (Bayesian
            Information Criterion), or AICc (Akaike Information Criterion
            corrected for finite sample sizes)

    Returns:
        np.ndarray: information criterion as 1D array (n_lambdas)

    Note: AIC and BIC calculations taken from scikit-learn's LarsCV

    """
    if coefs.ndim == 1:
        coefs = coefs[:, np.newaxis]

    n_samples = y.shape[0]

    criterion = criterion.lower()
    if criterion == 'aic' or criterion == 'aicc':
        K = 2
    elif criterion == 'bic':
        K = log(n_samples)
    else:
        raise ValueError('Criterion must be either AIC, BIC, or AICc')

    resid = y[:, np.newaxis] - np.dot(X, coefs)
    mse = np.mean(resid ** 2, axis=0)

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


def plot_paths(results, which_to_label=None):
    import matplotlib
    import matplotlib.pyplot as plt
    plt.clf()
    interactive_state = plt.isinteractive()
    xvalues = -np.log(results.lambdas[1:])
    for index, path in enumerate(results.coefficients):
        if which_to_label and results.indices[index] in which_to_label:
            if which_to_label[results.indices[index]] is None:
                label = "$x_{%d}$" % results.indices[index]
            else:
                label = which_to_label[results.indices[index]]
        else:
            label = None


        if which_to_label and label is None:
            plt.plot(xvalues, path[1:], ':')
        else:
            plt.plot(xvalues, path[1:], label=label)

    plt.xlim(np.amin(xvalues), np.amax(xvalues))

    if which_to_label is not None:
        plt.legend(loc='upper left')
    plt.title('Regularization paths ($\\rho$ = %.2f)' % results.balance)
    plt.xlabel('$-\log(\lambda)$')
    plt.ylabel('Value of regression coefficient $\hat{\\beta}_i$')
    plt.show()
    plt.interactive(interactive_state)
