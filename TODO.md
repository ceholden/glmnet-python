TODO:

1. [ ] Implement ElasticNet/Lasso CV & IC classes
    * [x] IC
    * [x] CV
2. [x] Test / ensure class is pickle-able using joblib
    * Yes
3. [x] Speed timing comparison sklearn vs glmnet wrapper
    * See `example/glmnet_demo.ipynb`
    * **TLDR**: `glmnet` Lasso can be 2-4.5x faster than `sklearn` Lasso
        * ~4.5 faster on "boston" dataset and ~2x faster on "diabetes" dataset
4. [x] Example IPython notebook
5. [x] Explain difference between this fork and other forks
6. [ ] Tests
