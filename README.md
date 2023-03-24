[![DOI](https://zenodo.org/badge/617341653.svg)](https://zenodo.org/badge/latestdoi/617341653)

# Introduction

This repository contains all material necessary to replicate the results in:

Traag, VA, M Malgarini, and S Sarlo. “Metrics and Peer Review Agreement at the Institutional Level,” [arXiv:2006.14830v2](https://arxiv.org/abs/2006.14830v2)

This README explains how to use the scripts in this repository to replicate the results. The basis of the analysis is the hierarchical Bayesian model, which is programmed in `stan`. The model itself is run from Python. All subsequent analyses of the results are done in separate Python scripts.

# Replication

The main results can be replicate as follows.

1. Obtain the VQR data from the Zenodo repository at https://doi.org/10.5281/zenodo.7759166.
   By default, the data is assumed to be stored in the `data` directory. You can change the directories in `common.py`.
2. Setup the replication environment using [`conda`](https://docs.conda.io/):
   ```
   conda env create -f environment.yml
   ```
   Instead of `conda` you can also use [`mamba`](https://mamba.readthedocs.io/).
3. Run `run_stan_models.py`.
   This compiles and runs the `stan` model in `model.stan`. Running all models can quite some time, since it has to be run for 11 different GEVs, for 4 different indicators, for 5 folds, testing both review and citation scores, totalling to 440 different models.
   **Note**: All scripts are assumed to be run from within the `src` directory.
4. Run `transform_fit_results.py`.
   This transforms the results from the various GEVs and all folds to a more convenient format for further analysis. Additionally, it also translates identifiers used in the `stan` model (which have to be consecutively numbered: 1, 2, ...) to identifiers from the actual VQR dataset.
5. Run analysis scripts:
   - `analyse_test_results.py`: produces the posterior test results, in particular the scatter plots of all predictions versus the observed scores (Figs. 4--7).
   - `analyse_train_results.py`: produces posterior predictive check results, and results for the parameter estimates (Figs. 8--10, S1, S2).
   - `analyse_MAD.py`:  produces the MA(P)D results and the distribution of the average absolute differences (Figs. 11--15).

There are number of results that do not depend on the `stan` model, and can be run separately. These are

- `analyse_post_hoc_stratification.py`: produces the post-hoc stratification analysis of the percentage of outputs per institution (Fig. 1).
- `analyse_missing.py`: produces the results of how many publications are missing reviewer scores in the various GEVs (Fig. 2).
- `model_illustrations.py`: produces the figures to illustrate the model, and to showcase the expected MAD for multiple reviewers (Fig. 3, Fig. 16).

If you want to experiment with the `stan` model, it might be useful to test the model separately, instead of having to run the `run_stan_models.py` entirely. You can do so using `test_stan_model.py`.
