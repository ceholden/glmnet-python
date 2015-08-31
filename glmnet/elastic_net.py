import numpy as np

from .glmnet import elastic_net
from .utils import mse_path, plot_paths


class ElasticNet(object):
    """ ElasticNet based on GLMNET

    Fit an elastic net model with a fixed L1/L2 penalty mixing
    parameter. Multiple ``lambdas`` may be fit in a given run, but
    this class has no useful method of cross-validating or assessing
    the best fit. If multiple ``lambdas`` are provided, model
    predictions or parameters will be based off of the ``lambda`` that
    yields the lowest mean squared error (MSE).

    Args:
        alpha (float): ElasticNet mixing parameter (0 <= alpha <= 1.0)
            specifying the mix of Ridge L2 (``alpha=0``) to Lasso L1
            (``alpha=1``) regularization.
        lambdas (iterable, or None): Constant that controls the degree
            of regularization. If None are specified, ``n_lambdas``
            number of lambdas are set automatically
        n_lambdas (int or None): If ``lambdas`` is None, ``n_lambdas``
            controls the number of automatically calculated ``lambdas``
        fit_intercept (bool): Fit intercept for the model. If False,
            data are already assumed to be centered.

    Attributes:
        coef_ (np.ndarray): 1D array of model coefficients
        intercept_ (float): intercept
        alpha (float): L1/L2 regularization mixing parameter
        lambda_ (float): value of lambda used when fitting model

    """
    def __init__(self, alpha=0.5, lambdas=None, n_lambdas=1,
                 fit_intercept=True):
        super(ElasticNet, self).__init__()
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.fit_intercept = fit_intercept

        self.coefs_, self.intercepts_, self.rsquareds_ = None, None, None

    def fit(self, X, y, weights=None):
        """ Fit a model predicting y from X design matrix

        Args:
            X (np.ndarray): 2D (n_obs x n_features) design matrix
            y (np.ndarray): 1D independent variable
            weights (np.ndarray): 1D array of weights for each
                observation in ``y``. If None, all observations are
                weighted equally

        Returns:
            object: return `self` with model results stored for method
                chaining

        """
        weights = weights or np.ones(y.shape[0])

        # Compute lambdas if necessary
        if self.lambdas is None:
            raise NotImplementedError('Have not added auto-calc of lambdas yet')

        # Fit elastic net
        n_lambdas, intercepts_, coefs_, ia, nin, rsquareds_, lambdas, _, jerr \
            = elastic_net(X, y, self.alpha,
                          lambdas=self.lambdas,
                          weights=weights)

        # ia is 1 indexed
        ia = np.trim_zeros(ia, 'b') - 1

        # glmnet.f returns coefficients as 'compressed' array that
        # requires re-indexing using ia and nin
        self.coefs_ = np.zeros_like(coefs_)
        self.coefs_[ia, :] = coefs_[:np.max(nin), :n_lambdas]
        self.intercepts_ = intercepts_
        self.rsquareds_ = rsquareds_

        # Calculate "best" lambda for prediction based on some criteria
        self._idx_best_lambda = self._calc_best_lambda(X, y)
        self.lambda_ = lambdas[self._idx_best_lambda]

        return self

    def predict(self, X):
        """ Predict yhat using model

        Args:
            X (np.ndarray): 2D (n_obs x n_features) design matrix

        Returns:
            np.ndarray: 1D yhat prediction

        """
        return (np.dot(X, self.coef_) + self.intercept_)

    @property
    def coef_(self):
        if self.coefs_ is not None:
            return self.coefs_[:, self._idx_best_lambda]

    @property
    def intercept_(self):
        if self.intercepts_ is not None:
            return self.intercepts_[self._idx_best_lambda]

    @property
    def rsquared_(self):
        if self.rsquareds_ is not None:
            return self.rsquareds_[self._idx_best_lambda]

    def _calc_best_lambda(self, X, y):
        mse = mse_path(self.coefs_, X, y)
        return np.argmin(mse)

    def __str__(self):
        coef = self.coef_
        n_nonzero = (np.abs(coef) > np.finfo(coef.dtype).eps).sum()
        return ("%s with %d non-zero coefficients (%.2f%%):\n"
                " * Intercept = %.7f\n"
                " * Alpha = %.7f\n"
                " * Lambda = %.7f\n"
                " * Training R^2: %.4f") % (self.__class__.__name__,
                                            n_nonzero,
                                            n_nonzero / float(len(coef)) * 100,
                                            self.intercept_,
                                            self.alpha,
                                            self.lambda_,
                                            self.rsquared_)


class Lasso(ElasticNet):
    """ Lasso based on GLMNET

    Fit a Lasso model. Multiple ``lambdas`` may be fit in a given run, but
    this class has no useful method of cross-validating or assessing
    the best fit. If multiple ``lambdas`` are provided, model
    predictions or parameters will be based off of the ``lambda`` that
    yields the lowest mean squared error (MSE).

    Args:
        lambdas (iterable, or None): Constant that controls the degree
            of regularization. If None are specified, ``n_lambdas``
            number of lambdas are set automatically
        n_lambdas (int or None): If ``lambdas`` is None, ``n_lambdas``
            controls the number of automatically calculated ``lambdas``
        fit_intercept (bool): Fit intercept for the model. If False,
            data are already assumed to be centered.

    Attributes:
        coef_ (np.ndarray): 1D array of model coefficients
        intercept_ (float): intercept
        alpha (float): L1/L2 regularization mixing parameter
        lambda_ (float): value of lambda used when fitting model

    """
    def __init__(self, lambdas=None, n_lambdas=1,
                 fit_intercept=True):
        super(Lasso, self).__init__(alpha=1.0,
                                    lambdas=lambdas, n_lambdas=n_lambdas,
                                    fit_intercept=fit_intercept)


def elastic_net_path(X, y, alpha=0.5, **kwargs):
    """Return full path for ElasticNet"""
    n_lambdas, intercepts, coefs, _, _, _, lambdas, _, jerr \
    = elastic_net(X, y, alpha, **kwargs)
    return lambdas, coefs, intercepts


def lasso_path(X, y, **kwargs):
    """ Return full path for Lasso"""
    return elastic_net_path(X, y, alpha=1.0, **kwargs)
