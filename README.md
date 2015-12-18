# glmnet-python
Python wrapper to the Fortran implementation of GLMNET by Friedman *et al.*

This is a fork of [`glmnet-python`](https://github.com/dwf/glmnet-python) by [David Warde-Farley](https://github.com/dwf) who is the original author of the majority of the wrapper code, especially all of the difficult parts! This fork has restructured the code to follow the [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) model estimation API while providing some additional capabilities, including cross-validation of `lambda` values.

## Comparison with other forks

As of December 18th, 2015, there is [another fork of `glmnet-python`](https://github.com/shuras/glmnet-python) by [Github user "shuras"](https://github.com/shuras) that I would recommend using in favor of this fork. This other fork has a wider range of capabilities and is more actively under development. A non-exhaustive list of capabilities offered by the ["shuras" *et al.* fork](https://github.com/shuras/glmnet-python) include:

* Plot and diagnostic utilities
* Logistic regression
* Handling of sparse data
* Multi-response elastic nets
* Tests and more examples
* Planned support of Cox and Poisson models

Basically, steer clear of this fork unless you want a fast replacement for running GLMNET within a `scikit-learn` API framework.

## Requirements

* `numpy>=1.3`
* `scikit-learn>=0.14.0`

## Building
In order to get double precision working without modifying Friedman's
code, some compiler trickery is required. The wrappers have been written
such that everything returned is expected to be a `real*8` i.e. a
double-precision floating point number, and unfortunately the code is
written in a way Fortran is often written with simply `real` specified,
letting the compiler decide on the appropriate width. `f2py` assumes
`real` are always 4 byte/ single precision, hence the manual change in
the wrappers to `real*8`, but that change requires the actual Fortran
code to be compiled with 8-byte reals, otherwise bad things will happen
(the stack will be blown, program will hang or segfault, etc.).

AFAIK, this package requires `gfortran` to build. `g77` will not work as
it does not support `-fdefault-real-8`.

The way to get this to build properly is:

    python setup.py config_fc --fcompiler=gnu95 \
        --f77flags='-fdefault-real-8' \
        --f90flags='-fdefault-real-8' build

The `--fcompiler=gnu95` business may be omitted if gfortran is the only
Fortran compiler you have installed, but the compiler flags are
essential.

## License
Friedman's code in `glmnet.f` is released under the GPLv2, necessitating
that any code that uses it (including my wrapper, and anyone using my
wrapper) be released under the GPLv2 as well. See LICENSE for details.

That said, to the extent that they are useful in the absence of the GPL
Fortran code (i.e. not very), my portions may be used under the 3-clause
BSD license.

## Thanks

* Thanks to David Warde-Farley for his original work on the Python wrapper that contributed 99% of the effort I've based my additions on.

From David Warde-Farley:
* To Jerome Friedman for the fantastically fast and efficient Fortran code.
* To Pearu Peterson for writing `f2py` and answering my dumb questions.
* To Dag Sverre Seljebotn for his help with `f2py` wrangling.
* To Kevin Jacobs for showing me [his wrapper](http://code.google.com/p/glu-genetics/source/browse/trunk/glu/lib/glm/glmnet.pyf) which helped me side-step some problems with the auto-generated `.pyf`.

## References

* J Friedman, T Hastie, R Tibshirani (2010). ["Regularization Paths for
  Generalized Linear Models via Coordinate
  Descent"](http://www.jstatsoft.org/v33/i01/paper).
* J Friedman, T Hastie, H Hofling, R Tibshirani (2007). ["Pathwise Coordinate
  Optimization"](http://arxiv.org/pdf/0708.1485.pdf").
