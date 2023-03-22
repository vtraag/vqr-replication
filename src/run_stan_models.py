#%%
from cmdstanpy import CmdStanModel
import os
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from common import unique_id, nuniq, group_kfold_partition, results_dir

#%% Define function for preparing the data

from typing import Literal
def prepare_data(paper_df, citation_score, fold, prediction_type: Literal['citation', 'review']):
  """
  Parameters
  ----------
  paper_df : pandas Dataframe
    Dataframe of all papers that should be fitted.
  citation_score : str
    The citation score to use in the model.
  fold : int
    The fold to use for splitting into train and test set.
  prediction_type : str
    The type of prediction that should be done.
    'citation' will use citations to predict peer review scores, using the prior
    estimated coefficients.
    'review' will use review scores to predict other peer review scores, using
    the prior estimated coefficients.

  Returns
  -------
  CmdStanMCMC fitted objects.

  """

  # Split into training and test data for specific fold
  train_df = paper_df[paper_df['fold'] != fold]
  test_df = paper_df[paper_df['fold'] == fold]

  # Create specific dataframe for the citation information
  citation_df = (
                  train_df[['new_paper_id', citation_score]]
                  .rename(columns={citation_score:
                                  'citation_score'})
                  )

  # Include citation data of test set also if we are using that for prediction
  if (prediction_type == 'citation'):
      citation_df = pd.concat([citation_df,
                              (test_df[['new_paper_id', citation_score]]
                                .rename(columns={citation_score: 'citation_score'})
                              )
                              ]
                            )

  # Drop missing citation scores and sort
  citation_df = (
                  citation_df
                  .dropna()
                  .sort_values('new_paper_id')
                )
    
  # Create specific dataframe for the review information
  # We use both reviews for training.
  review_df = pd.concat([train_df[['new_paper_id', 'REV_1_SCORE']]\
                          .rename(columns={'REV_1_SCORE': 'review_score'}),
                        train_df[['new_paper_id', 'REV_2_SCORE']]\
                          .rename(columns={'REV_2_SCORE': 'review_score'})],
                      axis=0,
                      ignore_index=True)

  # We only use review score 1 for testing.
  if (prediction_type == 'review'):
      review_df = pd.concat([review_df, 
                            test_df[['new_paper_id', 'REV_1_SCORE']]\
                            .rename(columns={'REV_1_SCORE': 'review_score'})],
                            axis=0,
                            ignore_index=True)

  review_df = review_df.dropna()
    
  # Drop missing review scores and sort
  review_df = ( 
                review_df
                .dropna()
                .sort_values('new_paper_id')
              )

  # Create the data dictionary for passing to Stan
  data = {
      'N_papers':    paper_df.shape[0],
      'N_institutions':   nuniq(paper_df['new_institution_id']),
      'institution_per_paper': paper_df['new_institution_id'],

      'N_review_scores':    review_df.shape[0],
      'review_score': (review_df['review_score'] - 2).astype('int'), # Should be between 1-28
      'paper_per_review_score': review_df['new_paper_id'],
      
      'N_citation_scores': citation_df.shape[0],
      'citation_score': citation_df['citation_score'],
      'paper_per_citation_score': citation_df['new_paper_id'],
      'citation_percentile_score': 'PERCENTILE' in citation_score,
  }

  return data

#%% Create output directory

output_dir = results_dir

#%% Load model

stan_file = os.path.abspath('./model.stan')
model = CmdStanModel(stan_file=stan_file)
print(model)

#%% Load data

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')
metric_df.index.name = 'original_paper_id'

#%%

n_chains = 4 # The number of chains to be used in the stan model

n_folds = 5

# The citation scores we will test
citation_scores = ['ncs', 
                   'njs',
                   'PERCENTILE_INDICATOR_VALUE', 
                   'PERCENTILE_CITATIONS']

# We will repeat all fits for each GEV
for GEV, GEV_df in metric_df.groupby('GEV_id'):
    
  # The stan model expects continuous integer identifiers for each paper and
  # institution. Since the general identifiers are unique for each
  # paper/institution in the entire dataset, we need to make new identifiers
  # for each subset that we test.

  # Create new unique institution ID if not yet present
  GEV_df['new_institution_id'] = unique_id(GEV_df[['INSTITUTION_ID']])

  # Create new unique paper ID if not yet present  
  GEV_df['new_paper_id'] = np.arange(GEV_df.shape[0]) + 1  

  # Create k-fold partition
  GEV_df['fold'] = group_kfold_partition(GEV_df['new_institution_id'], n_folds)

  # Write to CSV so that we are able to match the new paper / institution IDs 
  # back to the original IDs during the analysis of the results.
  (output_dir / GEV).mkdir(parents=True, exist_ok=True)
  GEV_df.to_csv(output_dir / GEV / 'papers.csv')
  
  # We separately fit all citation scores
  for citation_score in citation_scores:
  
    # We repeat the fit for each fold
    for fold in range(n_folds):

      # We predict both on citations and on review
      for prediction_type in ['citation', 'review']:

        # Set the overall dir for outputting all fit results
        fit_dir = output_dir / GEV / citation_score / prediction_type / f'{fold}'
        
        # Prepare the data
        data = prepare_data(GEV_df, citation_score, fold, prediction_type)
        
        # Fit the model
        fit = model.sample(data=data, chains=n_chains,
                           output_dir=fit_dir,
                           adapt_delta=0.9,
                           max_treedepth=20)
        
        # Save diagnosis to file
        diagnose_filename = Path(fit.runset.csv_files[0]).stem + '-diagnose.txt'
        with open(fit_dir / diagnose_filename, 'wt') as f:
          f.write(fit.diagnose())
