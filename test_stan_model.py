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
fit = model.sample(data=data, chains=1, output_dir = output_dir)


#%%

print(fit.diagnose())

#%%

draws_df = fit.draws_pd()

#%%
import seaborn as sns

sns.distplot(draws_df['scale_paper[2]'])
sns.distplot(draws_df['scale_inst[2]'])