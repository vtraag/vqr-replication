#%%
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from analysis_functions import extract_variable
import numpy as np
from common import gev_names_df

#%% Set the directory we want to transform the fit results for

results_dir = Path('../results/20230118233207')

#%% Load the original data

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')

metric_df = (metric_df
             .reset_index()
             .rename(columns={'index': 'paper_id'})
            )

metric_df = pd.merge(metric_df, gev_names_df, 
                     on='GEV_id', how='left')

metric_df = (metric_df
             .set_index('paper_id')
             .sort_index()
            )

metric_df['REV_SCORE'] = metric_df[['REV_1_SCORE', 'REV_2_SCORE']].mean(axis=1)

#%%

citation_scores = ['ncs', 
                   'njs',
                   'PERCENTILE_INDICATOR_VALUE', 
                   'PERCENTILE_CITATIONS']

for citation_score in citation_scores:
  
  citation_prediction_draws_df = pd.read_csv(results_dir / citation_score / 'citation_prediction' / 'draws.csv')
  review_prediction_draws_df = pd.read_csv(results_dir / citation_score / 'review_prediction' / 'draws.csv')
    
  output_dir = results_dir / 'figures' / citation_score / 'prediction'
  
  output_dir.mkdir(parents=True, exist_ok=True)

  #%% Combine both predictions
  
  citation_pred_df = extract_variable(citation_prediction_draws_df, 'review_score_ppc', axis='columns')
  review_pred_df = extract_variable(review_prediction_draws_df, 'review_score_ppc', axis='columns')
  
  pred_df = pd.concat([citation_pred_df.rename(columns={'review_score_ppc': 'citation_pred'}), 
                       review_pred_df.rename(columns={'review_score_ppc': 'review_pred'})],
                      axis=1)
  
  # Reshape the dataframe such that it has the paper identifiers
  # as the index, with the prediction types and the individuals draws
  # as the columns
  pred_df = (pred_df
             .T
             .unstack(level=0)
             .swaplevel(0, 1, axis='columns')
             )
  
  # Now make sure it is properly sorted
  pred_df = pred_df.sort_index(axis='index')
  pred_df = pred_df.sort_index(axis='columns')

  #%% Create summary dataframe
  
  def percentile(n):
    def percentile_(x):
      if not isinstance(x,pd.Series):
        raise ValueError('need Series argument')
      return np.percentile(x, n)
    percentile_.__name__ = f'percentile_{n}'
    return percentile_

  summary_pred_df = (pred_df
                     .T
                     .groupby(level=0)
                     .agg(['mean', 'std', percentile(2.5), percentile(97.5)])
                     .T
                     .unstack(level=1)
                     )
  #%% Plot results for review, observed vs. posterior

  plt_df = pd.merge(metric_df, summary_pred_df.loc[:,'citation_pred'], 
                    left_index=True, right_index=True)
  
  g = sns.relplot(plt_df, x='REV_2_SCORE', y='mean', 
                  col='GEV', col_order=gev_names_df['GEV'], col_wrap=4,
                  alpha=0.4)

  (g.set_axis_labels("Observed review score", "Predicted review score")
    .set_titles("{col_name}")
    .tight_layout(w_pad=0))

  plt.savefig(output_dir / 'citation_pred.pdf', bbox_inches='tight')
  plt.close()
  
  #%% Plot results for review, observed vs. posterior
  plt_df = pd.merge(metric_df, summary_pred_df.loc[:,'review_pred'], 
                    left_index=True, right_index=True)
  
  g = sns.relplot(plt_df, x='REV_2_SCORE', y='mean', 
                  col='GEV', col_order=gev_names_df['GEV'], col_wrap=4,
                  alpha=0.4)

  (g.set_axis_labels("Observed review score", "Predicted review score")
    .set_titles("{col_name}")
    .tight_layout(w_pad=0))

  plt.savefig(output_dir / 'review_pred.pdf', bbox_inches='tight')
  plt.close()  
  
  #%% Plot two predicted posteriors versus each other
  plt_df = pd.merge(metric_df, summary_pred_df, 
                    left_index=True, right_index=True)  
  g = sns.relplot(plt_df, 
                  x=('review_pred', 'mean'), 
                  y=('citation_pred', 'mean'), 
                  col='GEV', col_order=gev_names_df['GEV'], col_wrap=4,
                  alpha=0.4)
  
  (g.set_axis_labels("Review-based predicted review score", "Citation-based predicted review score")
    .set_titles("{col_name}")
    .tight_layout(w_pad=0))  
  
  plt.savefig(output_dir / 'review_pred_vs_citation_pred.pdf', bbox_inches='tight')
  
  #%% Calculated absolute difference from REV_2_SCORE
  
  abs_diff_df = np.abs(pred_df.subtract(metric_df['REV_2_SCORE'], axis='index'))
  
  #%% Calculate summary of absolute differences
  
  summary_abs_diff_df = (abs_diff_df
                         .T
                         .groupby(level=0)
                         .agg(['mean'])
                         .T
                         .unstack(level=1)
                         )  
  
  #%% Save abs diff summary to file
  
  summary_abs_diff_df.to_csv(results_dir / citation_score / 'abs_diff.csv')
    
  #%% Aggregate to institutional level
  
  inst_metric_df = metric_df.groupby(['INSTITUTION_ID', 'GEV']).mean()
 
  # Create institutional predictions
  inst_gev_df = metric_df[['INSTITUTION_ID', 'GEV']]
  metric_pred_df = pd.merge(inst_gev_df, pred_df, 
                           left_index=True, right_index=True)
  
  inst_pred_df = metric_pred_df.groupby(['INSTITUTION_ID', 'GEV']).mean()

  inst_pred_df.columns = pd.MultiIndex.from_tuples(c for c in inst_pred_df.columns)
  
  #%%
  
  summary_inst_pred_df = (inst_pred_df
                         .T
                         .groupby(level=0)
                         .agg(['mean', 'std', percentile(2.5), percentile(97.5)])
                         .T
                         .unstack(level=-1)
                         )
      
  #%% Plot results for review, observed vs. posterior

  plt_df = pd.merge(inst_metric_df, summary_inst_pred_df.loc[:,'citation_pred'], 
                    left_index=True, right_index=True)
  
  g = sns.relplot(plt_df, x='REV_2_SCORE', y='mean', 
                  col='GEV', col_order=gev_names_df['GEV'], col_wrap=4,
                  alpha=0.4)

  (g.set_axis_labels("Observed review score", "Predicted review score")
    .set_titles("{col_name}")
    .tight_layout(w_pad=0))

  plt.savefig(output_dir / 'inst_citation_pred.pdf', bbox_inches='tight')
  #plt.close()
  
  #%% Plot results for review, observed vs. posterior
  plt_df = pd.merge(inst_metric_df, summary_inst_pred_df.loc[:,'review_pred'], 
                    left_index=True, right_index=True)
  
  g = sns.relplot(plt_df, x='REV_1_SCORE', y='mean', 
                  col='GEV', col_order=gev_names_df['GEV'], col_wrap=4,
                  alpha=0.4)

  (g.set_axis_labels("Observed review score", "Predicted review score")
    .set_titles("{col_name}")
    .tight_layout(w_pad=0))

  plt.savefig(output_dir / 'inst_review_pred.pdf', bbox_inches='tight')
  #plt.close()  
  
  #%% Plot two predicted posteriors versus each other
  
  plt_df = pd.merge(inst_metric_df, summary_inst_pred_df, 
                    left_index=True, right_index=True)
  
  g = sns.relplot(plt_df, 
                  x=('review_pred', 'mean'), 
                  y=('citation_pred', 'mean'), 
                  col='GEV', col_order=gev_names_df['GEV'], col_wrap=4,
                  alpha=0.4)
  
  (g.set_axis_labels("Review-based predicted review score", "Citation-based predicted review score")
    .set_titles("{col_name}")
    .tight_layout(w_pad=0))  
  
  plt.savefig(output_dir / 'inst_review_pred_vs_citation_pred.pdf', bbox_inches='tight')
  
   
  #%% Calculated absolute difference from REV_2_SCORE
  
  inst_abs_diff_df = np.abs(inst_pred_df.subtract(inst_metric_df['REV_2_SCORE'], axis='index'))
  
  #%% Calculate summary of absolute differences
  
  inst_summary_abs_diff_df = (inst_abs_diff_df
                         .T
                         .groupby(level=0)
                         .agg(['mean'])
                         .T
                         .unstack(level=-1)
                         )  
  
  #%% Save abs diff summary to file
  
  inst_summary_abs_diff_df.to_csv(results_dir / citation_score / 'inst_abs_diff.csv')  
