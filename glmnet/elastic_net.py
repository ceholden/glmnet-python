import inspect

import numpy as np
import six

from .glmnet import elastic_net
from .utils import IC_path, mse_path


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
            (``alpha=1``) regularization (default: 0.5).
        lambdas (iterable, or None): Constant that controls the degree
            of regularization. If None are specified, ``n_lambdas``
            number of lambdas are set automatically (default: None).
        n_lambdas (int or None): If ``lambdas`` is None, ``n_lambdas``
            controls the number of automatically calculated ``lambdas``
            (default: 1).
        standardize (bool): standardize X before fitting (default: True)

    Attributes:
        coef_ (np.ndarray): 1D array of model coefficients
        intercept_ (float): intercept
        alpha (float): L1/L2 regularization mixing parameter
        lambda_ (float): value of lambda used when fitting model

    """
    _score_method = 'Mean Squared Error'

    def __init__(self,
                 alpha=0.5,
                 lambdas=None,
                 n_lambdas=1,
                 standardize=True):
        super(ElasticNet, self).__init__()
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.standardize = standardize

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
        # Fit elastic net
        kwargs = dict(weights=weights,
                      lambdas=self.lambdas,
                      nlam=self.n_lambdas)
        self.intercepts_, self.coefs_, self.rsquareds_, lambdas = \
            enet_path(X, y, alpha=self.alpha, **kwargs)

        # Calculate "best" lambda for prediction based on some criteria
        self.mse_path_ = self._score_lambda(X, y)
        self._idx_best_lambda = self._best_lambda(self.mse_path_)
        self.lambda_ = lambdas[self._idx_best_lambda]

        return self

    def predict(self, X):
        """ Predict yhat using model

        Args:
            X (np.ndarray): 2D (n_obs x n_features) design matrix

        Returns:
            np.ndarray: 1D yhat prediction

        """
        return np.dot(X, self.coef_) + self.intercept_

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

    def _score_lambda(self, X, y):
        return mse_path(X, y, self.coefs_, self.intercepts_)

    def _best_lambda(self, scores):
        return np.argmin(scores)

    def __str__(self):
        coef = self.coef_
        n_nonzero = (np.abs(coef) > np.finfo(coef.dtype).eps).sum()
        return ("%s with %d non-zero coefficients (%.2f%%):\n"
                " * Intercept = %.7f\n"
                " * Alpha = %.7f\n"
                " * Lambda = %.7f\n"
                " * Training R^2: %.4f") % (self.__class__.__name__, n_nonzero,
                                            n_nonzero / float(len(coef)) * 100,
                                            self.intercept_, self.alpha,
                                            self.lambda_, self.rsquared_)

    def plot(self):
        """ Plot regularization path and scores of ``lambdas``
        """
        try:
            import matplotlib.pyplot as plt
            plt.style.available
        except (ImportError, AttributeError):
            raise ImportError('Requires matplotlib>=1.4.0 for plotting')

        x_lambda, x_lambdas = -np.log10(self.lambda_), -np.log10(self.lambdas)

        with plt.style.context('ggplot'):
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)

            if self.mse_path_.ndim == 1:
                ax1.plot(x_lambdas, self.mse_path_, ls='--', lw=2)
            else:
                mean_score = np.mean(self.mse_path_, axis=1)
                std_score = np.std(self.mse_path_, axis=1)
                ax1.plot(x_lambdas, mean_score, ls='--', lw=2)
                ax1.errorbar(x_lambdas, mean_score, yerr=std_score,
                             fmt='', ls='', c='grey')

            ax1.axvline(x_lambda, lw=2, ls='--', color='k')
            ax1.set_ylabel('%s' % self._score_method)

            ax2.plot(x_lambdas, self.coefs_.T, ls='--', lw=2)
            ax2.axvline(x_lambda, lw=2, ls='--', color='k')
            ax2.set_xlabel('$-\log(\lambda)$')
            ax2.set_ylabel('Coefficient $\hat{\\beta}_i$')

            fig.suptitle('%s ($\\lambda = %s$)' % (self.__class__.__name__,
                                                   x_lambda),
                         fontsize='16')

            plt.tight_layout()
            plt.show()

    def get_params(self, deep=True):
        """ Return parameters for this estimator

        Args:
            deep (bool): return the parameters from parameters of this
                estimator that are also estimators

        Returns:
            dict: parameter names mapped to their values

        """
        # Get our own __init__ signature
        args, _, _, _ = inspect.getargspec(self.__init__)
        args.pop(0)  # remove `self` from __init__

        params = dict()
        for key in args:
            value = getattr(self, key, None)

            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                params.update((key + '__' + k, val) for k, val in deep_items)
            params[key] = value

        return params

    def set_params(self, **params):
        """ Set parameters for estimator

        Args:
            params (dict): dict of parameter=value to set

        Returns:
            self
        """
        if not params:
            return self

        _params = self.get_params()

        for key, value in six.iteritems(params):
            if key not in _params:
                raise ValueError('Invalid parameter %s for %s' %
                                 (key, self.__class__.__name__))
            setattr(self, key, value)

        return self


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
            number of lambdas are set automatically (default: None).
        n_lambdas (int or None): If ``lambdas`` is None, ``n_lambdas``
            controls the number of automatically calculated ``lambdas``
            (default: 1).
        standardize (bool): standardize X before fitting (default: True)

    Attributes:
        coef_ (np.ndarray): 1D array of model coefficients
        intercept_ (float): intercept
        alpha (float): L1/L2 regularization mixing parameter
        lambda_ (float): value of lambda used when fitting model

    """

    def __init__(self, lambdas=None, n_lambdas=1, standardize=True):
        super(Lasso, self).__init__(alpha=1.0,
                                    lambdas=lambdas,
                                    n_lambdas=n_lambdas,
                                    standardize=standardize)


class ElasticNetIC(ElasticNet):
    """ ElasticNet based on GLMNET with lambda selected by information criterion

    Fit an elastic net model with a fixed L1/L2 penalty mixing
    parameter. Multiple ``lambdas`` may be fit in a given run, but
    this class has no useful method of cross-validating or assessing
    the best fit. If multiple ``lambdas`` are provided, model
    predictions or parameters will be based off of the ``lambda`` that
    yields the minimum information criterion.

    Args:
        alpha (float): ElasticNet mixing parameter (0 <= alpha <= 1.0)
            specifying the mix of Ridge L2 (``alpha=0``) to Lasso L1
            (``alpha=1``) regularization (default: 0.5).
        lambdas (iterable, or None): Constant that controls the degree
            of regularization. If None are specified, ``n_lambdas``
            number of lambdas are set automatically (default: None).
        n_lambdas (int or None): If ``lambdas`` is None, ``n_lambdas``
            controls the number of automatically calculated ``lambdas``
            (default: 1).
        standardize (bool): standardize X before fitting (default: True)
        criterion (str): Information criterion used to select best ``lambdas``
            for fit. Available ICs are AIC, BIC, and AICc (default: AIC).

    Attributes:
        coef_ (np.ndarray): 1D array of model coefficients
        intercept_ (float): intercept
        alpha (float): L1/L2 regularization mixing parameter
        lambda_ (float): value of lambda used when fitting model

    """
    def __init__(self,
                 alpha=0.5,
                 lambdas=None,
                 n_lambdas=1,
                 standardize=True,
                 criterion='aic'):
        self._score_method = criterion
        super(ElasticNetIC, self).__init__(
            alpha=alpha,
            lambdas=lambdas,
            n_lambdas=n_lambdas,
            standardize=standardize)

    def _score_lambda(self, X, y):
        return IC_path(X, y, self.coefs_, self.intercepts_,
                       criterion=self.criterion)

    def _best_lambda(self, scores):
        return np.argmin(scores)


class LassoIC(ElasticNetIC):
    """ Lasso based on GLMNET with lambda selected by information criterion

    Fit a Lasso model. Multiple ``lambdas`` may be fit in a given run, but
    this class has no useful method of cross-validating or assessing
    the best fit. If multiple ``lambdas`` are provided, model
    predictions or parameters will be based off of the ``lambda`` that
    yields the lowest mean squared error (MSE).

    Args:
        lambdas (iterable, or None): Constant that controls the degree
            of regularization. If None are specified, ``n_lambdas``
            number of lambdas are set automatically (default: None).
        n_lambdas (int or None): If ``lambdas`` is None, ``n_lambdas``
            controls the number of automatically calculated ``lambdas``
            (default: 1).
        standardize (bool): standardize X before fitting (default: True)
        criterion (str): Information criterion used to select best ``lambdas``
            for fit. Available ICs are AIC (default), BIC, and AICc.

    Attributes:
        coef_ (np.ndarray): 1D array of model coefficients
        intercept_ (float): intercept
        alpha (float): L1/L2 regularization mixing parameter
        lambda_ (float): value of lambda used when fitting model

    """

    def __init__(self,
                 lambdas=None,
                 n_lambdas=1,
                 standardize=True,
                 criterion='aic'):
        self.criterion = criterion
        super(LassoIC, self).__init__(
            lambdas=lambdas,
            n_lambdas=n_lambdas,
            standardize=standardize)


def path_residuals(X, y, train, test, alpha=0.5, **kwargs):
    """ Fit on test and return MSE of all ``lamdbas`` in test samples

    Args:
        X (np.ndarray): 2D (n_obs x n_features) design matrix
        y (np.ndarray): 1D independent variable
        train (np.ndarray): indices of X/y for training model
        test (np.ndarray): indices of X/y for testing model and MSE calculation
        alpha (float): ElasticNet mixing parameter (0 <= alpha <= 1.0)
            specifying the mix of Ridge L2 (``alpha=0``) to Lasso L1
            (``alpha=1``) regularization (default: 0.5).
        kwargs: additional arguments provided to GLMNET

    Returns:
        np.ndarray: mean squared error (MSE) for each lambda specified in
            ``kwargs``

    """
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    intercepts, coefs, _, _ = enet_path(X_train, y_train,
                                        alpha=alpha, **kwargs)

    return mse_path(X_test, y_test, coefs, intercepts)


def enet_path(X, y, alpha=0.5, **kwargs):
    """ Convenience wrapper for running elastic_net that transforms coefs

    Args:
        X (np.ndarray): 2D (n_obs x n_features) design matrix
        y (np.ndarray): 1D independent variable
        alpha (float): ElasticNet mixing parameter (0 <= alpha <= 1.0)
            specifying the mix of Ridge L2 (``alpha=0``) to Lasso L1
            (``alpha=1``) regularization (default: 0.5).
        kwargs: additional arguments provided to GLMNET

    Returns:
        tuple: intercepts, coefficients, rsquareds, and lambdas as np.ndarrays

    """
    n_lambdas, intercepts_, coefs_, ia, nin, rsquareds_, lambdas, _, jerr \
        = elastic_net(X, y, alpha, **kwargs)

    # ia is 1 indexed
    ia = np.trim_zeros(ia, 'b') - 1

    # glmnet.f returns coefficients as 'compressed' array that
    # requires re-indexing using ia and nin
    coefs = np.zeros_like(coefs_)
    coefs[ia, :] = coefs_[:np.max(nin), :n_lambdas]

    return intercepts_, coefs, rsquareds_, lambdas
