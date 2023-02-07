#%%
from cmdstanpy import CmdStanModel
import os
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from common import unique_id, nuniq

#%% Define function for preparing the data

from typing import Literal
def prepare_data(paper_df, citation_score, prediction_type: Literal['none', 'citation', 'review']='none', prior_summary_df=None):
  """
  Parameters
  ----------
  paper_df : pandas Dataframe
    Dataframe of all papers that should be fitted.
  citation_score : str
    The citation score to use in the model.
  prediction_type : str
    The type of prediction that should be done.
    'none' will not make any prediction, and just fit all available data.
    'citation' will use citations to predict peer review scores, using the prior
    estimated coefficients.
    'review' will use review scores to predict other peer review scores, using
    the prior estimated coefficients.
  prior_fit: CmdStanMCMC
    Prior fit of coefficients that was run with the prediction_type set to 'none'.

  Returns
  -------
  CmdStanMCMC fitted objects.

  """
  
  # The stan model expects continuous integer identifiers for each paper and
  # institution. Since the general identifiers are unique for each
  # paper/institution in the entire dataset, we need to make new identifiers
  # for each subset that we test.

  # Create new unique institution ID if not yet present
  if 'new_institution_id' not in paper_df.columns:
    paper_df['new_institution_id'] = unique_id(paper_df[['INSTITUTION_ID']])

  # Create new unique paper ID if not yet present  
  if 'new_paper_id' not in paper_df.columns:
    paper_df['new_paper_id'] = np.arange(paper_df.shape[0]) + 1  

  # Create the data dictionary for passing to Stan
  data =  {
            'N_papers':    paper_df.shape[0],
            'N_institutions':   nuniq(paper_df['new_institution_id']),
            'institution_per_paper': paper_df['new_institution_id'],
          }
  
  if (prediction_type in ['none', 'citation']):
    # Create specific dataframe for the citation information
    citation_df = (
                    paper_df[['new_paper_id', citation_score]]
                    .rename(columns={citation_score:
                                    'citation_score'})
                    .dropna()
                  )
    
    # Add to data for stan model
    data |= {
              'N_citation_scores': citation_df.shape[0],
              'citation_score': citation_df['citation_score'],
              'paper_per_citation_score': citation_df['new_paper_id'],
              'citation_percentile_score': 'PERCENTILE' in citation_score,
            }
  else:
    data |= {
              'N_citation_scores': 0,
              'citation_score': [],
              'paper_per_citation_score': [],
              'citation_percentile_score': 0,
            }

  if (prediction_type in ['none', 'review']):

    # Create specific dataframe for the review information
    if (prediction_type == 'none'):
      # We use both reviews when not predicting anything, but just
      # fitting the overall model.
      review_df = pd.concat([paper_df[['new_paper_id', 'REV_1_SCORE']]\
                                .rename(columns={'REV_1_SCORE': 'review_score'}),
                            paper_df[['new_paper_id', 'REV_2_SCORE']]\
                                .rename(columns={'REV_2_SCORE': 'review_score'})],
                            axis=0,
                            ignore_index=True)
    else:
      # We only use review score 1 when predicting review score 2.
      review_df = (
                    paper_df[['new_paper_id', 'REV_1_SCORE']]
                    .rename(columns={'REV_1_SCORE': 'review_score'})
                  )
    
    # Drop missing review scores and sort
    review_df = ( 
                  review_df
                  .dropna()
                  .sort_values('new_paper_id')
                )
    
    # Add to data for stan model
    data |= {
              'N_review_scores':    review_df.shape[0],
              'review_score': (review_df['review_score'] - 2).astype('int'), # Should be between 1-28
              'paper_per_review_score': review_df['new_paper_id'],
            }
  else:
    data |= {
              'N_review_scores': 0,
              'review_score': [],
              'paper_per_review_score': [],
            }

  if (prediction_type == 'none'):
    # We do not want to predict review scores, we just want to fit
    # the overall model.
    data |= {
              'use_estimated_priors': 0,

              # The below are estimated coefficients from other models.
              'sigma_paper_value_mu': 0,
              'sigma_paper_value_sigma': 1,

              # Coefficient of citation
              'beta_mu': 0,
              'beta_sigma': 1,

              # Standard deviation of citation
              'sigma_cit_mu': 0,
              'sigma_cit_sigma': 1,

              # Standard deviation of peer review.
              'sigma_review_mu': 0,
              'sigma_review_sigma': 1,

              # Nonzero citation parameters
              'alpha_nonzero_cit_mu': 0,
              'alpha_nonzero_cit_sigma': 1,              

              'beta_nonzero_cit_mu': 0,
              'beta_nonzero_cit_sigma': 1
            }
  else:
    # We now want to predict review scores, based either on a single review
    # score or on a citation score. This means that we will use the estimates
    # that we estimated on the full model.
    data |= {
              'use_estimated_priors': 1,

              # The below are estimated coefficients from other models.
              'sigma_paper_value_mu': prior_summary_df.loc['sigma_paper_value', 'Mean'],
              'sigma_paper_value_sigma': prior_summary_df.loc['sigma_paper_value', 'StdDev'],

              # Coefficient of citation
              'beta_mu': prior_summary_df.loc['beta', 'Mean'],
              'beta_sigma': prior_summary_df.loc['beta', 'StdDev'],

              # Standard deviation of citation
              'sigma_cit_mu': prior_summary_df.loc['sigma_cit', 'Mean'],
              'sigma_cit_sigma': prior_summary_df.loc['sigma_cit', 'StdDev'],

              # Standard deviation of peer review.
              'sigma_review_mu': prior_summary_df.loc['sigma_review', 'Mean'],
              'sigma_review_sigma': prior_summary_df.loc['sigma_review', 'StdDev'],

              # Nonzero citation parameters
              'alpha_nonzero_cit_mu': prior_summary_df.loc['alpha_nonzero_cit', 'Mean'],
              'alpha_nonzero_cit_sigma': prior_summary_df.loc['alpha_nonzero_cit', 'StdDev'],              

              'beta_nonzero_cit_mu': prior_summary_df.loc['beta_nonzero_cit', 'Mean'],
              'beta_nonzero_cit_sigma': prior_summary_df.loc['beta_nonzero_cit', 'StdDev']
            }
  return data

#%% Create output directory

now = dt.datetime.now().strftime("%Y%m%d%H%M%S")
output_dir = Path(f'../results/{now}')

#%% Load model

stan_file = os.path.abspath('./model.stan')
model = CmdStanModel(stan_file=stan_file)
print(model)

#%% Load data

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')

#%%

chains = 4 # The number of chains to be used in the stan model

# The citation scores we will test
citation_scores = ['ncs', 
                   'njs',
                   'PERCENTILE_INDICATOR_VALUE', 
                   'PERCENTILE_CITATIONS']

# We will repeat all fits for each GEV
for GEV, GEV_df in metric_df.groupby('GEV_id'):
  
  for citation_score in citation_scores:
  
    # Set the overall dir for outputting all fit results
    fit_dir = output_dir / GEV / citation_score
    
    #%% Prior fit
    
    # First prepare the data
    data = prepare_data(GEV_df, citation_score)
    
    # Then fit the overall model
    fit = model.sample(data=data, chains=chains,
                       output_dir = fit_dir / 'prior',
                       adapt_delta=0.9,
                       max_treedepth=20)
    
    # Get summary of fit for later use
    prior_summary_df = fit.summary()
    
    # Save diagnosis to file
    diagnose_filename = Path(fit.runset.csv_files[0]).stem + '-diagnose.txt'
    with open(fit_dir / 'prior' / diagnose_filename, 'wt') as f:
      f.write(fit.diagnose())
    
    #%% Predict based on citation scores
    
    # First prepare the data, making use of the previously fitted model
    data = prepare_data(GEV_df, citation_score, prediction_type='citation', 
                        prior_summary_df=prior_summary_df )
  
    # Fit the model and predict review scores
    fit = model.sample(data=data, chains=chains,
                       output_dir = fit_dir / 'citation_prediction',
                       adapt_delta=0.9,
                       max_treedepth=20)
    
    # Save diagnosis to file
    diagnose_filename = Path(fit.runset.csv_files[0]).stem + '-diagnose.txt'
    with open(fit_dir / 'citation_prediction' / diagnose_filename, 'wt') as f:
      f.write(fit.diagnose())
        
    #%% Predict based on review scores
    
    # First prepare the data, making use of the previously fitted model    
    data = prepare_data(GEV_df, citation_score, prediction_type='review', 
                        prior_summary_df=prior_summary_df )
    
    # Fit the model and predict review scores    
    fit = model.sample(data=data, chains=chains,
                       output_dir = fit_dir / 'review_prediction',
                       adapt_delta=0.9,
                       max_treedepth=20)
    
    # Save diagnosis to file
    diagnose_filename = Path(fit.runset.csv_files[0]).stem + '-diagnose.txt'
    with open(fit_dir / 'review_prediction' / diagnose_filename, 'wt') as f:
      f.write(fit.diagnose())

  GEV_df.to_csv(output_dir / GEV / 'papers.csv')
