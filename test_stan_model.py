#%%
from cmdstanpy import CmdStanModel
import os
import pandas as pd
import numpy as np
import datetime as dt
#%%

stan_file = os.path.abspath('./model.stan')
model = CmdStanModel(stan_file=stan_file)
print(model)

#%%

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')

#%%

def unique_id(df, start_id=1):
    """ Create unique identifiers for each unique row in df."""
    return df.groupby(list(df.columns)).ngroup() + start_id

def nuniq(df):
    return df.drop_duplicates().shape[0]

paper_df = (metric_df
            .query('GEV_id == "4"')\
               [['INSTITUTION_ID',
                 'REV_1_SCORE',
                 'REV_2_SCORE',
                 'ncs']]
            .dropna()
            .sort_values('INSTITUTION_ID')
           )
paper_df['new_institution_id'] = unique_id(paper_df[['INSTITUTION_ID']])
paper_df['new_paper_id'] = np.arange(paper_df.shape[0]) + 1

paper_df['REV_SCORE'] = (paper_df['REV_1_SCORE'] + paper_df['REV_2_SCORE'])/2

review_df = (
                pd.concat([paper_df[['new_paper_id', 'REV_1_SCORE']]\
                           .rename(columns={'REV_1_SCORE': 'review_score'}),
                       paper_df[['new_paper_id', 'REV_2_SCORE']]\
                           .rename(columns={'REV_2_SCORE': 'review_score'})],
                      axis=0,
                      ignore_index=True)
                .sort_values('new_paper_id')    
            )

#%%

data = {
    'N_reviews':    review_df.shape[0],        
    'N_papers':    paper_df.shape[0],
    'N_institutions':   nuniq(paper_df['new_institution_id']),
    'review_score': (review_df['review_score'] - 2).astype('int'), # Should be between 1-28
    'paper_per_review': review_df['new_paper_id'],
    'citation_score': paper_df['ncs'],
    'institution_per_paper': paper_df['new_institution_id']
}

#%%
now = dt.datetime.now().strftime("%Y%m%d%H%M%S")
output_dir = f'../results/{now}'
fit = model.sample(data=data, chains=1, output_dir = output_dir, adapt_delta=0.99)

#%%

print(fit.diagnose())

#%%

draws_df = fit.draws_pd()

summary_df = fit.summary()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(draws_df['beta'])

#%%

sns.distplot(draws_df['sigma_cit'])
sns.distplot(draws_df['sigma_review'])

#%%
sns.pairplot(draws_df[['sigma_cit', 
                       'sigma_review']])

#%%
def extract_variable(df, variable):
  """ Extracts a variable from a summary dataframe from stan, and vectorizes
  the indices. That is, if there is a variable x[1], x[2], etc.. in a stan
  dataframe, it is extracted by this function to 
  id1    x
  1      value
  2
  """
  
  import re
  
  # Find relevant indices (rows in this case)
  var_re = re.compile(f'{variable}\[[\d,]*\]')
  variable_index = [idx for idx in df.index if var_re.match(idx) is not None]
  
  # Split the name of each matching index, i.e.
  # variable[1,2] will be split into 1, 2
  split_variable_indices = [idx[len(variable)+1:-1].split(',')
                                 for idx in variable_index]
  
  variable_df = df.loc[variable_index,]
  
  for i, idx in enumerate(zip(*split_variable_indices)):
    variable_df[f'index{i}'] = list(map(int, idx))
  
  return variable_df
  
#%%

review_score_ppc_df = extract_variable(summary_df, 'review_score_ppc')

plt.plot(paper_df['REV_SCORE'], review_score_ppc_df['Mean'], '.')

plt.xlabel('Observed review score')
plt.ylabel('Posterior predicted review score')

#%%

citation_ppc_df = extract_variable(summary_df, 'citation_ppc')

plt.plot(paper_df['ncs'], citation_ppc_df['Mean'], '.')

plt.xlabel('Observed citation score')
plt.ylabel('Posterior predicted citation score')
