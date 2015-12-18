import numpy as np

from . import _glmnet

_DEFAULT_THRESH = 1.0e-4
_DEFAULT_FLMIN = 0.001
_DEFAULT_NLAM = 100


def elastic_net(predictors, target, balance,
                memlimit=None, largest=None, thr=_DEFAULT_THRESH,
                weights=None, penalties=None, standardize=True, exclude=None,
                lambdas=None, flmin=None, nlam=None,
                overwrite_pred_ok=False, overwrite_targ_ok=True):
    """ Raw-output wrapper for elastic net linear regression.

    Args:
        predictors (np.ndarray): X design matrix
        target (np.ndarray): y dependent variables
        balance (float): Family member index (0 is ridge, 1 is Lasso)
        memlimit (int): Maximum number of variables allowed to enter all models
            along path
        largest (int): Maximum number of variables allowed to enter largest
            model
        thr (float): Minimum change in largest coefficient
        weights (np.ndarray): Relative weighting per observation case
        penalties (np.ndarray): Relative penalties per predictor (0 = no
            penalty)(vp in Fortran code)
        standardize (bool): Standardize input variables before proceeding?
            (isd in Fortran code)
        exclude (np.ndarray): Predictors to exclude altogether from fitting
            (jd in Fortran code)
        lambdas (np.ndarray): User specified lambda values (ulam in Fortran
            code). Do not specify ``lambdas`` if ``flmin`` or ``nlam`` are also
            provided.
        flmin (float): Fraction of largest lambda at which to stop
            (default: 0.001)
        nlam (int): The (maximum) number of lambdas to try (default: 100)
        overwrite_pred_ok (bool): Allow overwriting of X
        overwrite_targ_ok (bool): Allow overwriting of y
    """
    # Decide on largest allowable models for memory/convergence.
    memlimit = predictors.shape[1] if memlimit is None else memlimit
    # If largest isn't specified use memlimit.
    largest = memlimit if largest is None else largest
    if memlimit < largest:
        raise ValueError('Need largest <= memlimit')

    if exclude is not None:
        # Add one since Fortran indices start at 1
        exclude += 1
        jd = np.array([len(exclude)] + exclude)
    else:
        jd = np.zeros(1)

    if lambdas is not None and flmin is not None:
        raise ValueError("Can't specify both lambdas & flmin keywords")
    elif lambdas is not None:
        lambdas = np.atleast_1d(lambdas)
        flmin = 2.  # Pass flmin > 1.0 indicating to use the user-supplied.
        nlam = len(lambdas)
    else:
        lambdas = None
        flmin = _DEFAULT_FLMIN
        nlam = _DEFAULT_NLAM

    # If predictors is a Fortran contiguous array, it will be overwritten.
    # Decide whether we want this. If it's not Fortran contiguous it will
    # be copied into that form anyway so there's no chance of overwriting.
    if np.isfortran(predictors):
        if not overwrite_pred_ok:
            # Might as well make it F-ordered to avoid ANOTHER copy.
            predictors = predictors.copy(order='F')
    # Target being a 1-dimensional array will usually be overwritten
    # with the standardized version unless we take steps to copy it.
    if not overwrite_targ_ok:
        target = target.copy()

    # Uniform weighting if no weights are specified.
    if weights is None:
        weights = np.ones(predictors.shape[0])

    # Uniform penalties if none were specified.
    if penalties is None:
        penalties = np.ones(predictors.shape[1])

    # Call the Fortran wrapper.
    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr =  \
        _glmnet.elnet(balance, predictors, target, weights, jd, penalties,
                      memlimit, flmin, lambdas, thr,
                      nlam=nlam, isd=standardize)

    # Check for errors, documented in glmnet.f.
    if jerr != 0:
        if jerr == 10000:
            raise ValueError('cannot have max(vp) < 0.0')
        elif jerr == 7777:
            raise ValueError('all used predictors have 0 variance')
        elif jerr < 7777:
            raise MemoryError('elnet() returned error code %d' % jerr)
        else:
            raise Exception('unknown error: %d' % jerr)

    return lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr
