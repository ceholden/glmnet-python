import numpy as np

from . import _glmnet

_DEFAULT_THRESH = 1.0e-4
_DEFAULT_FLMIN = 0.001
_DEFAULT_NLAM = 100


def elastic_net(predictors, target, balance, memlimit=None,
                largest=None, **kwargs):
    """
    Raw-output wrapper for elastic net linear regression.
    """

    # Mandatory parameters
    predictors = np.asanyarray(predictors)
    target = np.asanyarray(target)

    # Decide on largest allowable models for memory/convergence.
    memlimit = predictors.shape[1] if memlimit is None else memlimit

    # If largest isn't specified use memlimit.
    largest = memlimit if largest is None else largest

    if memlimit < largest:
        raise ValueError('Need largest <= memlimit')

    # Flags determining overwrite behavior
    overwrite_pred_ok = False
    overwrite_targ_ok = False

    thr = _DEFAULT_THRESH   # Minimum change in largest coefficient
    weights = None          # Relative weighting per observation case
    vp = None               # Relative penalties per predictor (0 = no penalty)
    isd = True              # Standardize input variables before proceeding?
    jd = np.zeros(1)        # Predictors to exclude altogether from fitting
    ulam = None             # User-specified lambda values
    flmin = _DEFAULT_FLMIN  # Fraction of largest lambda at which to stop
    nlam = _DEFAULT_NLAM    # The (maximum) number of lambdas to try.

    for keyword in kwargs:
        if keyword == 'overwrite_pred_ok':
            overwrite_pred_ok = kwargs[keyword]
        elif keyword == 'overwrite_targ_ok':
            overwrite_targ_ok = kwargs[keyword]
        elif keyword == 'threshold':
            thr = kwargs[keyword]
        elif keyword == 'weights':
            if np.all(kwargs.get(keyword)):
                weights = np.asarray(kwargs[keyword]).copy()
        elif keyword == 'penalties':
            vp = kwargs[keyword].copy()
        elif keyword == 'standardize':
            isd = bool(kwargs[keyword])
        elif keyword == 'exclude':
            # Add one since Fortran indices start at 1
            exclude = (np.asarray(kwargs[keyword]) + 1).tolist()
            jd = np.array([len(exclude)] + exclude)
        elif keyword == 'lambdas':
            if 'flmin' in kwargs:
                raise ValueError("Can't specify both lambdas & flmin keywords")
            ulam = np.atleast_1d(kwargs[keyword])
            flmin = 2.  # Pass flmin > 1.0 indicating to use the user-supplied.
            nlam = len(ulam)
        elif keyword == 'flmin':
            flmin = kwargs[keyword]
            ulam = None
        elif keyword == 'nlam':
            if kwargs.get('lambdas') is not None:
                continue  # let `lambdas` override nlam
            nlam = kwargs[keyword]
        else:
            raise ValueError("Unknown keyword argument '%s'" % keyword)

    # If predictors is a Fortran contiguous array, it will be overwritten.
    # Decide whether we want this. If it's not Fortran contiguous it will
    # be copied into that form anyway so there's no chance of overwriting.
    if np.isfortran(predictors):
        if not overwrite_pred_ok:
            # Might as well make it F-ordered to avoid ANOTHER copy.
            predictors = predictors.copy(order='F')

    # target being a 1-dimensional array will usually be overwritten
    # with the standardized version unless we take steps to copy it.
    if not overwrite_targ_ok:
        target = target.copy()

    # Uniform weighting if no weights are specified.
    if weights is None:
        weights = np.ones(predictors.shape[0])

    # Uniform penalties if none were specified.
    if vp is None:
        vp = np.ones(predictors.shape[1])

    # Call the Fortran wrapper.
    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr =  \
        _glmnet.elnet(balance, predictors, target, weights, jd, vp,
                      memlimit, flmin, ulam, thr, nlam=nlam, isd=isd)

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
