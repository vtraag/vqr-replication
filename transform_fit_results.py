#%%
import cmdstanpy
import os
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
#%% Set the directory we want to transform the fit results for

results_dir = Path('../results/20230118233207')

#%% Load the original data

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')

#%% Define functions

def get_fit(GEV, citation_score, prediction_type):
  """
  Get the fit for a particular GEV, citation score and prediction type.

  Parameters
  ----------
  GEV : str
    The GEV id for which to retrieve the fit results.
  citation_score : str
    The citation score for which to retrieve the fit results.
  prediction_type : str ['prior', 'citation_prediction', 'review_prediction']
    The type of scores to retrieve.

  Returns
  -------
  paper_df : pd.Dataframe
    The paper dataframe which contains the new identifiers used by stan.
  fit : CmdStanMCMC
    The fit object from stan
  """
  paper_df = pd.read_csv(results_dir / GEV / 'papers.csv')
  paper_df = paper_df.rename(columns={'Unnamed: 0': 'original_paper_id'})
  paper_df = paper_df.set_index('new_paper_id')
  
  fit = cmdstanpy.from_csv(results_dir / GEV / citation_score / prediction_type)
  
  return paper_df, fit

def get_institution_id_link(paper_df):
  """
  Get the institutional links. Note this uses the global variable metric_df to 
  link the original papers back to the institutional identifiers (which are
  not available from paper_df).

  Parameters
  ----------
  paper_df : pd.Dataframe
    The dataframe containing both the new and the original paper identifiers.

  Returns
  -------
  inst_id_df : pd.Dataframe
    The dataframe containing the new and the original institutional identifiers,
    with the new identifiers as the index.

  """
  inst_id_df = (
                  pd.merge(
                    paper_df[['new_institution_id', 'original_paper_id']],
                    metric_df,
                    left_on='original_paper_id',
                    right_index=True)\
                  [['INSTITUTION_ID', 'new_institution_id']]
                  .drop_duplicates()
                  .sort_values('INSTITUTION_ID')
                  .set_index('new_institution_id')
               )
  return inst_id_df

def rename_columns(draws_df, inst_id_df, GEV):
  """
  This functions renames the columns from the draws_df.
  
  Each paper-level variable will be reindexed so as to use the original paper
  identifier. Each insitutional-level variable will be reindexed so as to use
  the original institutional identifier. All other estimates will be provided
  with a GEV as an index (e.g. 'sigma_paper_value[4b]'). Default stan columns
  (e.g. 'lp__') are excluded.

  Parameters
  ----------
  draws_df : pd.Dataframe
    The draws as obtained from the stan fit.
  inst_id_df : pd.Dataframer
    The institutional link, see get_institution_id_link.
  GEV : str
    The GEV for which these draws were obtained.

  Returns
  -------
  pd.Dataframe
    draws_df with renamed, limited, columns.

  """
    
  from parse import parse
  
  columns = draws_df.columns
  new_column_name = {}
  
  for column in columns:
    if any(name in column 
            for name in ['citation_ppc', 'review_score_ppc', 'value_per_paper']):
      
      # We reindex paper-based variables with the original paper identifier
      name, new_paper_id = parse('{}[{}]', column)
      original_paper_id = paper_df.loc[int(new_paper_id), 'original_paper_id']
      new_column_name[column] = f'{name}[{original_paper_id}]'
  
    elif 'value_inst' in column:
      
      # We reindex institution-based variables with the original institutional identifier
      name, new_inst_id = parse('{}[{}]', column)
      original_inst_id = inst_id_df.loc[int(new_inst_id), 'INSTITUTION_ID']
      new_column_name[column] = f'{name}[{original_inst_id}]'
      
    elif not column.endswith('__'):
      
      new_column_name[column] = f'{column}[{GEV}]'

  return draws_df.rename(columns=new_column_name)[new_column_name.values()]

#%%

gev_names_df = pd.DataFrame(
  [['1', "Mathematics and Computer Sciences"],
  ['2', "Physics"],
  ['3', "Chemistry"],
  ['4', "Earth Sciences"],
  ['5', "Biology"],
  ['6', "Medicine"],
  ['7', "Agricultural and veterinary sciences"],
  ['8b', "Civil Engineering"],
  ['9', "Industrial and Information Engineering"],
  ['11b', "Psychology"],
  ['13', "Economics and Statistics"]], columns=['GEV_id', 'GEV'])

#%%

citation_scores = ['ncs', 
                   'njs',
                   'PERCENTILE_INDICATOR_VALUE', 
                   'PERCENTILE_CITATIONS']

prediction_types = ['prior',
                    'citation_prediction',
                    'review_prediction']

for citation_score in citation_scores:
  for prediction_type in prediction_types:
    
    # We want to collect all draws from all models from all GEVs
    all_draws_df = []
    for row, (GEV_id, GEV_name) in gev_names_df.iterrows():
      
      # Get the original fit
      paper_df, fit = get_fit(GEV_id, citation_score, prediction_type)
      
      # Obtain the link from new to original institutional identifiers
      inst_id_df = get_institution_id_link(paper_df)
    
      if prediction_type == 'prior':  
        draws_df = fit.draws_pd()
      else:
        # If we do prediction, we are only interested in the predicted
        # review scores
        draws_df = fit.draws_pd('review_score_ppc')
        
      # Add the draws, with renamed columns
      all_draws_df.append(
        rename_columns(draws_df, inst_id_df, GEV_id)
        )
      
    # Concatenate all draws
    all_draws_df = pd.concat(all_draws_df, axis=1)
    
    # Save all draws
    output_dir = results_dir / citation_score / prediction_type
    output_dir.mkdir(parents=True, exist_ok=True)

    all_draws_df.to_csv(output_dir / 'draws.csv')
