import warnings

import numpy as np
try:
    from sklearn.externals.joblib import Parallel, delayed
    from sklearn.cross_validation import check_cv
except:
    raise ImportError('scikit-learn must be installed for cross-validation')

from .elastic_net import ElasticNet, Lasso, path_residuals


class ElasticNetCV(ElasticNet):
    """ ElasticNet based on GLMNET with lambda selected by cross-validation

    Fit an elastic net model with a fixed L1/L2 penalty mixing
    parameter. Multiple ``lambdas`` may be fit in a given run. If multiple
    ``lambdas`` are provided, model predictions or parameters will be based
    off of the ``lambda`` that yields the lowest mean squared error as
    determined by cross-validation

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
        cv (int or cross-validation generator): Specify how cross-validation
            should be performed. If an integer is passed, cross-validation
            generator will be a simple KFold with ``cv`` folds. Other
            cross-validation generators may also be passed (see
            ``sklearn.cross_validation`` for other generators) (default: 3).
        n_jobs (int): Number of CPUs to use during cross-validation. If ``-1``,
            use all CPUs (default: 1).

    Attributes:
        coef_ (np.ndarray): 1D array of model coefficients
        intercept_ (float): intercept
        alpha (float): L1/L2 regularization mixing parameter
        lambda_ (float): value of lambda used when fitting model
        mse_path_ (np.ndarray): mean squared error for each ``lambdas`` for
            each cross-validation fold

    """

    _score_method = 'Mean Squared Error'

    def __init__(self,
                 alpha=0.5,
                 lambdas=None,
                 n_lambdas=100,
                 standardize=True,
                 cv=3,
                 n_jobs=1):
        self.cv = cv
        self.n_jobs = n_jobs
        super(ElasticNetCV, self).__init__(
            alpha=alpha,
            lambdas=lambdas,
            n_lambdas=n_lambdas,
            standardize=standardize)

    def fit(self, X, y, weights=None, penalties=None, **kwargs):
        """ Fit a model predicting y from X design matrix

        Args:
            X (np.ndarray): 2D (n_obs x n_features) design matrix
            y (np.ndarray): 1D independent variable
            weights (np.ndarray): 1D array of weights for each
                observation in ``y``. If None, all observations are
                weighted equally
            penalties (np.ndarray, or None): 1D array of penalty weights for
                each feature in X. If None, all features are penalized
                equivalently. Features given a ``0`` penalty will not be
                penalized at all.
            kwargs (dict, optional): additional keyword arguments provided to
                ``glmnet.glmnet.elastic_net`` function

        Returns:
            object: return `self` with model results stored for method
                chaining

        """
        # Use `sklearn` to guarantee `cv` is a generator
        # Note: as of 0.16.1 `check_cv` raises warning because of refactoring
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            cv = check_cv(self.cv, X=X, y=y, classifier=False)
        folds = list(cv)

        kwargs.update({
            'lambdas': self.lambdas,
            'nlam': self.n_lambdas,
            'weights': weights,
            'penalties': penalties
        })
        jobs = (delayed(path_residuals)(X, y, train, test,
                                        alpha=self.alpha, **kwargs)
                for train, test in folds)
        mse_paths = Parallel(n_jobs=self.n_jobs, backend="threading")(jobs)
        self.mse_path_ = np.rollaxis(np.reshape(mse_paths, (len(folds), -1)),
                                     1, 0)
        mean_mse = np.mean(self.mse_path_, axis=1)
        best_lambda = np.argmin(mean_mse)

        # Fit model with best parameters
        if self.alpha == 1.0:
            model = Lasso()
        else:
            model = ElasticNet()

        common_params = dict((name, value)
                             for name, value in self.get_params().items()
                             if name in model.get_params())
        model.set_params(**common_params)
        model.lambdas = self.lambdas
        model.fit(X, y, **kwargs)

        # Update `self` with model results
        self.coefs_ = model.coefs_
        self.intercepts_ = model.intercepts_
        self.rsquareds_ = model.rsquareds_

        self._idx_best_lambda = best_lambda
        self.lambda_ = self.lambdas[best_lambda]

        return self


class LassoCV(ElasticNetCV):
    """ Lasso based on GLMNET with lambda selected by cross-validation

    Fit a Lasso model. Multiple ``lambdas`` may be fit in a given run. If
    multiple ``lambdas`` are provided, model predictions or parameters will be
    based off of the ``lambda`` that yields the lowest mean squared error as
    determined by cross-validation

    Args:
        lambdas (iterable, or None): Constant that controls the degree
            of regularization. If None are specified, ``n_lambdas``
            number of lambdas are set automatically (default: None).
        n_lambdas (int or None): If ``lambdas`` is None, ``n_lambdas``
            controls the number of automatically calculated ``lambdas``
            (default: 1).
        standardize (bool): standardize X before fitting (default: True)
        cv (int or cross-validation generator): Specify how cross-validation
            should be performed. If an integer is passed, cross-validation
            generator will be a simple KFold with ``cv`` folds. Other
            cross-validation generators may also be passed (see
            ``sklearn.cross_validation`` for other generators) (default: 3).
        n_jobs (int): Number of CPUs to use during cross-validation. If ``-1``,
            use all CPUs (default: 1).

    Attributes:
        coef_ (np.ndarray): 1D array of model coefficients
        intercept_ (float): intercept
        alpha (float): L1/L2 regularization mixing parameter
        lambda_ (float): value of lambda used when fitting model
        mse_path_ (np.ndarray): mean squared error for each ``lambdas`` for
            each cross-validation fold

    """
    def __init__(self,
                 lambdas=None,
                 n_lambdas=100,
                 standardize=True,
                 cv=3,
                 n_jobs=1):
        self.cv = cv
        self.n_jobs = n_jobs
        super(ElasticNetCV, self).__init__(
            alpha=1.0,
            lambdas=lambdas,
            n_lambdas=n_lambdas,
            standardize=standardize)
