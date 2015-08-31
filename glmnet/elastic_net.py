import numpy as np

from .glmnet import elastic_net


class ElasticNet(object):
    """ ElasticNet based on GLMNET

    Args:
        alpha (float): ElasticNet mixing parameter (0 <= alpha <= 1.0)
            specifying the mix of Ridge L2 (alpha=0) to Lasso L1 (alpha=1)
            regularization.
        lambdas (iterable, or None): Constant that controls the degree of
            regularization. If None are specified, `n_lambdas` number of
            lambdas are set automatically using `_alpha_grid` from scikit-learn
        n_lambdas (int or None): If ``lambdas`` is None, ```n_lambdas`` controls
            the number of automatically calculated ``lambdas``

    """
    def __init__(self, alpha=0.5, lambdas=None, n_lambdas=1):
        super(ElasticNet, self).__init__()
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas

        self.coef_ = None
        self.rsquared_ = None

    def fit(self, X, y, weights=None):
        """ Fit a model predicting y independent variable from X design matrix

        Args:
            X (np.ndarray): 2D design matrix
            y (np.ndarray): 1D independent variable
            weights (np.ndarray): 1D array of weights for each observation in
                `y`. If None, all observations are weighted equally

        Returns:
            ElasticNet: return `self` with model results stored for method
                chaining

        """
        self.weights = weights or np.ones(y.shape[0])

        # Compute lambdas if necessary
        if lambdas is None:
            raise NotImplementedError('Have not added auto-calc of lambdas yet')

        # Fit elastic net
        n_lambdas, intercept_, coef_, ia, nin, rsquared_, lambdas, _, jerr \
            = elastic_net(X, y, self.alpha,
                          lambdas=self.lambdas,
                          weights=weights)

        # ia is 1 indexed
        ia = np.trim_zeros(ia, 'b') - 1

        # glmnet.f returns coefficients as 'compressed' array that requires
        # re-indexing using ia and nin
        self.coef_ = np.zeros_like(coef_)
        self.coef_[ia, :] = coef_[:np.max(nin), :n_lambdas]
        self.intercept_ = intercept_
        self.rsquared_ = rsquared_

        # TODO: Calculate "best" alpha for prediction based on MSE

        return self

    def predict(self, X):
        # TODO: predict for best model
        return np.dot(X, self.coef_) + self.intercept_

    def __str__(self):
        n_non_zeros = (np.abs(self.coef_) != 0).sum()
        return ("%s with %d non-zero coefficients (%.2f%%)\n" + \
                " * Intercept = %.7f, Lambda = %.7f\n" + \
                " * Training r^2: %.4f") % \
                (self.__class__.__name__, n_non_zeros,
                 n_non_zeros / float(len(self.coef_)) * 100,
                 self.intercept_[0], self.alpha, self.rsquared_[0])


def elastic_net_path(X, y, rho, **kwargs):
    """Return full path for ElasticNet"""
    n_lambdas, intercepts, coefs, _, _, _, lambdas, _, jerr \
    = elastic_net(X, y, rho, **kwargs)
    return lambdas, coefs, intercepts

def Lasso(alpha):
    """Lasso based on GLMNET"""
    return ElasticNet(alpha, rho=1.0)

def lasso_path(X, y, **kwargs):
    """return full path for Lasso"""
    return elastic_net_path(X, y, rho=1.0, **kwargs)
