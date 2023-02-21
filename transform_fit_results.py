#%%
import cmdstanpy
import pandas as pd
from pathlib import Path
from common import gev_names_df

#%% Set the directory we want to transform the fit results for

results_dir = Path('../results/20230220131637')

#%% Load the original data

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')

#%% Define functions

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

def rename_columns_and_split(draws_df, inst_id_df, paper_df, fold, GEV):
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
  train_columns = []
  test_columns = []

  test_papers = set(paper_df[paper_df['fold'] == fold]['original_paper_id'])
  test_institutions = set(paper_df[paper_df['fold'] == fold]['INSTITUTION_ID'])

  for column in columns:
    if any(name in column
            for name in ['citation_ppc', 'review_score_ppc', 'value_per_paper']):

      # We reindex paper-based variables with the original paper identifier
      name, new_paper_id = parse('{}[{}]', column)
      original_paper_id = paper_df.loc[int(new_paper_id), 'original_paper_id']
      new_col = f'{name}[{original_paper_id}]'
      new_column_name[column] = new_col

      if original_paper_id in test_papers:
        test_columns.append(new_col)
      else:
        train_columns.append(new_col)

    elif 'value_inst' in column:

      # We reindex institution-based variables with the original institutional identifier
      name, new_inst_id = parse('{}[{}]', column)
      original_inst_id = inst_id_df.loc[int(new_inst_id), 'INSTITUTION_ID']
      new_col = f'{name}[{original_inst_id}]'
      new_column_name[column] = new_col

      if original_inst_id in test_institutions:
        test_columns.append(new_col)
      else:
        train_columns.append(new_col)

    elif not column.endswith('__'):

      new_col = f'{column}[{GEV}]'
      new_column_name[column] = new_col

      train_columns.append(new_col)

  renamed_df = draws_df.rename(columns=new_column_name)
  train_df = renamed_df[train_columns]
  test_df = renamed_df[test_columns]

  return train_df, test_df

#%%

citation_scores = ['ncs',
                   'njs',
                   'PERCENTILE_INDICATOR_VALUE',
                   'PERCENTILE_CITATIONS']

prediction_types = ['citation',
                    'review']

n_folds = 5

for citation_score in citation_scores:
  for prediction_type in prediction_types:

    # We want to collect all draws from all models from all GEVs
    train_draws_df = []
    test_draws_df = []

    for row, (GEV_id, GEV_name) in gev_names_df.iterrows():

      # Read papers for GEV
      paper_df = pd.read_csv(results_dir / GEV_id / 'papers.csv', index_col='new_paper_id')
      paper_df = paper_df.sort_index()

      # Collect draws from all folds
      train_folds_df = []
      test_folds_df = []

      for fold in range(n_folds):

        # Get the original fit
        fit = cmdstanpy.from_csv(results_dir / GEV_id / citation_score / prediction_type / f'{fold}')

        # Obtain the link from new to original institutional identifiers
        inst_id_df = get_institution_id_link(paper_df)

        # Get the draws from the fit
        draws_df = fit.draws_pd()

        # Rename the columns using the original paper IDs and institutional ID
        # Simultaneously split into test and training dataset.
        train_df, test_df = rename_columns_and_split(draws_df, inst_id_df, paper_df, fold, GEV_id)

        # We thin the training sample, because they will be included in n_folds - 1 training
        # so as to obtain in total the same number of draws for the training set as for the test set.
        n_sample_size = (int)(train_df.shape[0]/(n_folds - 1))
        train_folds_df.append(train_df.sample(n=n_sample_size).reset_index(drop=True))

        test_folds_df.append(test_df)

      #%% We concatenate all training draws, while ignoring the index
      train_df = pd.concat(train_folds_df, axis='index', ignore_index=True)

      # Since each element is included once in the test set and n_fold times
      # in the training set, this dataset contains NAs for the times an
      # element wasn't in the training set. This is not what is relevant to us
      # so we drop all those NAs per columns. Subsequently, we drop all NAs
      # which gets rid of the rows at the bottom. Note that this also drops
      # some sample draws for parameters that are always present in all folds
      # (e.g. the various alpha, beta, and sigma values)
      train_df = train_df.apply(lambda x: pd.Series(x.dropna().values)).dropna()

      train_draws_df.append(train_df)

      #%% We want to concatenate the various columns for the test data
      test_df = pd.concat(test_folds_df, axis='columns')

      test_draws_df.append(test_df)
    
    #%%
    # Concatenate all draws from across GEVs
    all_test_draws_df = pd.concat(test_draws_df, axis=1)
    all_train_draws_df = pd.concat(train_draws_df, axis=1)

    # Save all draws
    output_dir = results_dir / citation_score / prediction_type
    output_dir.mkdir(parents=True, exist_ok=True)

    all_test_draws_df.to_csv(output_dir / 'test_draws.csv')
    all_train_draws_df.to_csv(output_dir / 'train_draws.csv')
