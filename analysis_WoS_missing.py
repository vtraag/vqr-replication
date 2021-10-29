import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import re
import statsmodels.formula.api as smf
import os
from datetime import date

colors = sns.palettes.mpl_palette('Set1', 8)

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

today = date.today().strftime('%Y%m%d')

#%%
np.random.seed(0)
min_n_per_institution = 1

results_dir = '../results/min={min}/missing'.format(min=min_n_per_institution)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

#%% Read data
df = pd.read_csv('../data/public/metrics.csv')
df = pd.merge(gev_names_df, df, on='GEV_id')
df = df.set_index(['INSTITUTION_ID', 'GEV'])

# Remove items with missing metrics
df = df[~pd.isna(df['ncs'])]

df['GEV_numeric'] = df['GEV_id'].apply(lambda x: int(re.match(r'\d*', x)[0]))
#%% Add missing lower and upper bound values
min_rev_score = 3
max_rev_score = 30
    
for col in ['REV_1_SCORE', 'REV_2_SCORE']:
  df[f'{col}_lb'] = df[col].fillna(min_rev_score)
  df[f'{col}_ub'] = df[col].fillna(max_rev_score)  

#%% Overview per GEV
df['missing_either'] = pd.isna(df['REV_1_SCORE']) | pd.isna(df['REV_2_SCORE'])
df['missing_both'] = pd.isna(df['REV_1_SCORE']) & pd.isna(df['REV_2_SCORE'])
df['n_reviewers'] = pd.isna(df['REV_1_SCORE']).astype(int) + pd.isna(df['REV_2_SCORE']).astype(int)
GEV_n_pubs = df.groupby(level=['GEV'], sort=False)\
                      .aggregate({'REV_1_SCORE': 'count', 'REV_2_SCORE': 'count',
                                  'missing_either': 'sum', 'missing_both': 'sum'})
GEV_n_pubs = GEV_n_pubs.rename(columns={'REV_1_SCORE': 'n_pubs_rev1', 
                           'REV_2_SCORE': 'n_pubs_rev2'})
GEV_n_pubs['n_pubs'] = df.groupby(level=['GEV']).size()
GEV_n_pubs['n_pubs_missing_rev1'] = GEV_n_pubs['n_pubs'] - GEV_n_pubs['n_pubs_rev1']
GEV_n_pubs['n_pubs_missing_rev2'] = GEV_n_pubs['n_pubs'] - GEV_n_pubs['n_pubs_rev2']
GEV_n_pubs['prop_pubs_missing_rev1'] = GEV_n_pubs['n_pubs_missing_rev1']/GEV_n_pubs['n_pubs']
GEV_n_pubs['prop_pubs_missing_rev2'] = GEV_n_pubs['n_pubs_missing_rev2']/GEV_n_pubs['n_pubs']
GEV_n_pubs['prop_missing_both'] = GEV_n_pubs['missing_both']/GEV_n_pubs['n_pubs']
GEV_n_pubs['prop_missing_either'] = GEV_n_pubs['missing_either']/GEV_n_pubs['n_pubs']

#%%
sns.set_style('white')
sns.set_palette('Set1')
plt.figure(figsize=(6, 4))
xlabels = GEV_n_pubs.index
x = np.arange(len(xlabels))
gap = 0.01
group_gap = 0.3
width = 0.8
bars = []

h = 0
y = GEV_n_pubs['n_pubs'] - GEV_n_pubs['missing_both'];
b = plt.bar(x=x, height=y, bottom=h, width=width, label='2 reviewers')
h += y
y = GEV_n_pubs['missing_either']
b = plt.bar(x=x, height=y, bottom=h, width=width, label='1 reviewer')
h += y
y = GEV_n_pubs['missing_both']
b = plt.bar(x=x, height=y, bottom=h, width=width, label='No reviewer')

plt.xticks(np.arange(len(xlabels)), xlabels, rotation='vertical')
#plt.xlabel('GEV')
plt.ylabel('Number of publications')
plt.grid(axis='y', ls=':')
plt.legend(bbox_to_anchor=(0.5, 1.02), loc='center',
           ncol=5)
sns.despine()
plt.savefig(os.path.join(results_dir, 'GEV_missing_pubs.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(results_dir, 'GEV_missing_pubs.png'), dpi=300, bbox_inches='tight')

#%%
GEV_ind = df.reset_index().groupby(['GEV', 'n_reviewers'], sort=False)\
                      .mean()

#%% Calculate institutional sizes
n_per_institution = df.groupby(level=['INSTITUTION_ID', 'GEV'], sort=False)\
                      .aggregate({'REV_1_SCORE': 'count', 'REV_2_SCORE': 'count'})
n_per_institution.columns = ['n_pubs_rev1', 'n_pubs_rev2']
n_per_institution['n_pubs'] = df.groupby(level=['INSTITUTION_ID', 'GEV']).size()
n_per_institution['n_pubs_missing_rev1'] = n_per_institution['n_pubs'] - n_per_institution['n_pubs_rev1']
n_per_institution['n_pubs_missing_rev2'] = n_per_institution['n_pubs'] - n_per_institution['n_pubs_rev2']
limited_df = pd.merge(df.reset_index(), n_per_institution[n_per_institution['n_pubs'] >= min_n_per_institution].reset_index(), on=['INSTITUTION_ID', 'GEV'])
  
#%% Defin {max,min}_diff functions

def max_diff(x, y_lb, y_ub):
  assert((y_lb <= y_ub).all())
  d = np.maximum(np.abs(x - y_lb), 
                 np.abs(x - y_ub))
  return d

def min_diff(x, y_lb, y_ub):
  assert((y_lb <= y_ub).all())
  d = np.minimum(np.abs(x - y_lb), 
                 np.abs(x - y_ub))
  d[(x >= y_lb) & (x <= y_ub)] = 0
  return d

def max_diff2d(x_lb, x_ub, y_lb, y_ub):
  assert((x_lb <= x_ub).all())
  assert((y_lb <= y_ub).all())
  d_x_lb = np.maximum(np.abs(x_lb - y_lb), np.abs(x_lb - y_ub))
  d_x_ub = np.maximum(np.abs(x_ub - y_lb), np.abs(x_ub - y_ub))
  d = np.maximum(d_x_lb, d_x_ub)
  return d

def min_diff2d(x_lb, x_ub, y_lb, y_ub):
  assert((x_lb <= x_ub).all())
  assert((y_lb <= y_ub).all())
  d_x_lb = np.minimum(np.abs(x_lb - y_lb), np.abs(x_lb - y_ub))
  d_x_ub = np.minimum(np.abs(x_ub - y_lb), np.abs(x_ub - y_ub))
  d = np.minimum(d_x_lb, d_x_ub)
  d[(y_ub >= x_lb) & (y_lb <= x_ub)] = 0
  return d
    
def abs_perc_diff(x, y):
  d = np.abs(x - y)/y
  #d[pd.isna(d)] = 0
  return d

def max_perc_diff(x, y_lb, y_ub):
  assert((y_lb <= y_ub).all())  
  d = np.maximum(np.abs(x - y_lb)/y_lb,
                 np.abs(x - y_ub)/y_ub)
  return d;

def min_perc_diff(x, y_lb, y_ub):
  assert((y_lb <= y_ub).all())
  d = np.minimum(np.abs(x - y_lb)/y_lb, 
                 np.abs(x - y_ub)/y_ub)
  d[(x >= y_lb) & (x <= y_ub)] = 0
  return d

def max_perc_diff2d(x_lb, x_ub, y_lb, y_ub):
  assert((x_lb <= x_ub).all())
  assert((y_lb <= y_ub).all())
  d_x_lb = np.maximum(np.abs(x_lb - y_lb)/y_lb, np.abs(x_lb - y_ub)/y_ub)
  d_x_ub = np.maximum(np.abs(x_ub - y_lb)/y_lb, np.abs(x_ub - y_ub)/y_ub)
  d = np.maximum(d_x_lb, d_x_ub)
  return d

def min_perc_diff2d(x_lb, x_ub, y_lb, y_ub):
  assert((x_lb <= x_ub).all())
  assert((y_lb <= y_ub).all())
  d_x_lb = np.minimum(np.abs(x_lb - y_lb)/y_lb, np.abs(x_lb - y_ub)/y_ub)
  d_x_ub = np.minimum(np.abs(x_ub - y_lb)/y_lb, np.abs(x_ub - y_ub)/y_ub)
  d = np.minimum(d_x_lb, d_x_ub)
  d[(y_ub >= x_lb) & (y_lb <= x_ub)] = 0
  return d

#%% Calculate MAD and MAPD on institutional level
def calc_MAD_and_MAPD_inst(df):
  ##%% Aggregate to institute
  inst_df = df.groupby(['INSTITUTION_ID', 'GEV_id', 'GEV_numeric'])\
                      .aggregate({'REV_1_SCORE': 'mean',
                                  'REV_1_SCORE_lb': 'mean',
                                  'REV_1_SCORE_ub': 'mean',
                                  'REV_2_SCORE': 'mean',
                                  'REV_2_SCORE_lb': 'mean',
                                  'REV_2_SCORE_ub': 'mean',
                                  'ncs': 'mean',
                                  'njs': 'mean',
                                  'PERCENTILE_CITATIONS': 'mean',
                                  'PERCENTILE_INDICATOR_VALUE': 'mean',
                                  'n_pubs': 'mean',
                                  'n_pubs_missing_rev1': 'mean',
                                  'n_pubs_missing_rev2': 'mean'})

  MADs = []
  MAPDs = []
  for GEV_id, GEV_df in inst_df.groupby('GEV_id'):
    ##%% Predict values
    res_ncs = smf.quantreg(formula='REV_1_SCORE ~ ncs', data=GEV_df).fit(q=0.5, max_iter=10000)
    res_njs = smf.quantreg(formula='REV_1_SCORE ~ njs', data=GEV_df).fit(q=0.5, max_iter=10000)

    if GEV_id != '13':
      res_perc_cit = smf.quantreg(formula='REV_1_SCORE ~ PERCENTILE_CITATIONS', data=GEV_df).fit(q=0.5, max_iter=10000)
      res_perc_ind = smf.quantreg(formula='REV_1_SCORE ~ PERCENTILE_INDICATOR_VALUE', data=GEV_df).fit(q=0.5, max_iter=10000)

    GEV_df['pred_ncs'] = res_ncs.predict(GEV_df['ncs'])
    GEV_df['pred_njs'] = res_njs.predict(GEV_df['njs'])
    if GEV_id != '13':
      GEV_df['pred_perc_cit'] = res_perc_cit.predict(GEV_df['PERCENTILE_CITATIONS']) # To account for NA
      GEV_df['pred_perc_ind'] = res_perc_ind.predict(GEV_df['PERCENTILE_INDICATOR_VALUE']) # To account for NA
    else:
      GEV_df['pred_perc_cit'] = np.nan
      GEV_df['pred_perc_ind'] = np.nan

    GEV_df['abs_diff_ncs'] = np.abs(GEV_df['pred_ncs'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_njs'] = np.abs(GEV_df['pred_njs'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_perc_cit'] = np.abs(GEV_df['pred_perc_cit'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_perc_ind'] = np.abs(GEV_df['pred_perc_ind'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_rev_2'] = np.abs(GEV_df['REV_2_SCORE'] - GEV_df['REV_1_SCORE'])
    
    ##%% Obtain upper bounds with maximum possible absolute difference
    GEV_df['abs_diff_ncs_ub'] = max_diff(GEV_df['pred_ncs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_njs_ub'] = max_diff(GEV_df['pred_njs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_perc_cit_ub'] = max_diff(GEV_df['pred_perc_cit'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_perc_ind_ub'] = max_diff(GEV_df['pred_perc_ind'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_rev_2_ub'] = max_diff2d(GEV_df['REV_2_SCORE_lb'], GEV_df['REV_2_SCORE_ub'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
 
    ##%% Obtain lower bounds with minimum possible absolute differences
    GEV_df['abs_diff_ncs_lb'] = min_diff(GEV_df['pred_ncs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_njs_lb'] = min_diff(GEV_df['pred_njs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_perc_cit_lb'] = min_diff(GEV_df['pred_perc_cit'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_perc_ind_lb'] = min_diff(GEV_df['pred_perc_ind'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_rev_2_lb'] = min_diff2d(GEV_df['REV_2_SCORE_lb'], GEV_df['REV_2_SCORE_ub'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    ##%% Summarise absolute differences
    MAD = GEV_df.median()[['abs_diff_ncs', 'abs_diff_njs', 'abs_diff_perc_cit', 'abs_diff_perc_ind', 'abs_diff_rev_2',
                           'abs_diff_ncs_lb', 'abs_diff_njs_lb', 'abs_diff_perc_cit_lb', 'abs_diff_perc_ind_lb', 'abs_diff_rev_2_lb',
                           'abs_diff_ncs_ub', 'abs_diff_njs_ub', 'abs_diff_perc_cit_ub', 'abs_diff_perc_ind_ub', 'abs_diff_rev_2_ub']]
    
    MAD['GEV_id'] = GEV_id
    MADs.append(MAD)

    ##%% Calculate absolute differences
    cols = ['REV_1_SCORE', 'REV_1_SCORE_lb', 'REV_1_SCORE_ub',
            'REV_2_SCORE', 'REV_2_SCORE_lb', 'REV_2_SCORE_ub',
            'ncs', 'njs',
            'pred_ncs', 'pred_njs', 'pred_perc_cit', 'pred_perc_ind']

    GEV_df = GEV_df[cols + ['n_pubs']].copy()
    GEV_df[cols] = GEV_df[cols].mul(GEV_df['n_pubs'], axis=0)

    GEV_df['abs_perc_diff_ncs'] = abs_perc_diff(GEV_df['pred_ncs'], GEV_df['REV_1_SCORE'])
    GEV_df['abs_perc_diff_njs'] = abs_perc_diff(GEV_df['pred_njs'], GEV_df['REV_1_SCORE'])
    GEV_df['abs_perc_diff_perc_cit'] = abs_perc_diff(GEV_df['pred_perc_cit'], GEV_df['REV_1_SCORE'])
    GEV_df['abs_perc_diff_perc_ind'] = abs_perc_diff(GEV_df['pred_perc_ind'], GEV_df['REV_1_SCORE'])
    GEV_df['abs_perc_diff_rev_2'] = abs_perc_diff(GEV_df['REV_2_SCORE'], GEV_df['REV_1_SCORE'])

    ##%% Obtain upper bounds with maximum possible absolute percentage difference
    GEV_df['abs_perc_diff_ncs_ub'] = max_perc_diff(GEV_df['pred_ncs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_perc_diff_njs_ub'] = max_perc_diff(GEV_df['pred_njs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_perc_diff_perc_cit_ub'] = max_perc_diff(GEV_df['pred_perc_cit'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_perc_diff_perc_ind_ub'] = max_perc_diff(GEV_df['pred_perc_ind'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_perc_diff_rev_2_ub'] = max_perc_diff2d(GEV_df['REV_2_SCORE_lb'], GEV_df['REV_2_SCORE_ub'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
 
    ##%% Obtain lower bounds with minimum possible absolute percentage differences
    GEV_df['abs_perc_diff_ncs_lb'] = min_perc_diff(GEV_df['pred_ncs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_perc_diff_njs_lb'] = min_perc_diff(GEV_df['pred_njs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_perc_diff_perc_cit_lb'] = min_perc_diff(GEV_df['pred_perc_cit'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_perc_diff_perc_ind_lb'] = min_perc_diff(GEV_df['pred_perc_ind'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_perc_diff_rev_2_lb'] = min_perc_diff2d(GEV_df['REV_2_SCORE_lb'], GEV_df['REV_2_SCORE_ub'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    
    ##% Summarise absolute percentage differences
    MAPD = 100*GEV_df.median()[['abs_perc_diff_ncs', 'abs_perc_diff_njs', 'abs_perc_diff_perc_cit', 'abs_perc_diff_perc_ind', 'abs_perc_diff_rev_2',
                                'abs_perc_diff_ncs_ub', 'abs_perc_diff_njs_ub', 'abs_perc_diff_perc_cit_ub', 'abs_perc_diff_perc_ind_ub', 'abs_perc_diff_rev_2_ub',
                                'abs_perc_diff_ncs_lb', 'abs_perc_diff_njs_lb', 'abs_perc_diff_perc_cit_lb', 'abs_perc_diff_perc_ind_lb', 'abs_perc_diff_rev_2_lb']]
    MAPD['GEV_id'] = GEV_id
    MAPDs.append(MAPD)
  ##%% Create dataframes
  MAD = pd.concat(MADs, axis=1).T.set_index('GEV_id')
  MAPD = pd.concat(MAPDs, axis=1).T.set_index('GEV_id')
  ##%%
  return MAD, MAPD

#%% run MAD and MAPD calculations on institutional level
MAD, MAPD = calc_MAD_and_MAPD_inst(limited_df)

MAD = MAD.sort_index(axis=1)
MAD = pd.merge(gev_names_df, MAD, left_on='GEV_id', right_index=True)

MAPD = MAPD.sort_index(axis=1)
MAPD = pd.merge(gev_names_df, MAPD, left_on='GEV_id', right_index=True)

#%% Calculate MAD on individual level
def calc_MAD_ind(df):
  ##%%
  MADs = []
  for GEV_id, GEV_df in df.groupby('GEV_id'):
    ##%% Predict values
    res_ncs = smf.quantreg(formula='REV_1_SCORE ~ ncs', data=GEV_df).fit(q=0.5, max_iter=10000)
    res_njs = smf.quantreg(formula='REV_1_SCORE ~ njs', data=GEV_df).fit(q=0.5, max_iter=10000)
    
    if GEV_id != '13':
      res_perc_cit = smf.quantreg(formula='REV_1_SCORE ~ PERCENTILE_CITATIONS', data=GEV_df).fit(q=0.5, max_iter=10000)
      res_perc_ind = smf.quantreg(formula='REV_1_SCORE ~ PERCENTILE_INDICATOR_VALUE', data=GEV_df).fit(q=0.5, max_iter=10000)
    
    ##%% Calculate differences
    GEV_df['pred_ncs'] = res_ncs.predict(GEV_df['ncs'])
    GEV_df['pred_njs'] = res_njs.predict(GEV_df['njs'])

    if GEV_id != '13':
      GEV_df['pred_perc_cit'] = res_perc_cit.predict(GEV_df['PERCENTILE_CITATIONS']) # To account for NA
      GEV_df['pred_perc_ind'] = res_perc_ind.predict(GEV_df['PERCENTILE_INDICATOR_VALUE']) # To account for NA
    else:
      GEV_df['pred_perc_cit'] = np.nan
      GEV_df['pred_perc_ind'] = np.nan
      
    GEV_df['abs_diff_ncs'] = np.abs(GEV_df['pred_ncs'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_njs'] = np.abs(GEV_df['pred_njs'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_perc_cit'] = np.abs(GEV_df['pred_perc_cit'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_perc_ind'] = np.abs(GEV_df['pred_perc_ind'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_rev_2'] = np.abs(GEV_df['REV_2_SCORE'] - GEV_df['REV_1_SCORE'])

    # Obtain upper bounds with maximum possible absolute difference
    GEV_df['abs_diff_ncs_ub'] = max_diff(GEV_df['pred_ncs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_njs_ub'] = max_diff(GEV_df['pred_njs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_perc_cit_ub'] = max_diff(GEV_df['pred_perc_cit'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_perc_ind_ub'] = max_diff(GEV_df['pred_perc_ind'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_rev_2_ub'] = max_diff2d(GEV_df['REV_2_SCORE_lb'], GEV_df['REV_2_SCORE_ub'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub']) 
    
    ##%% Obtain lower bounds with minimum possible absolute differences
      
    GEV_df['abs_diff_ncs_lb'] = min_diff(GEV_df['pred_ncs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_njs_lb'] = min_diff(GEV_df['pred_njs'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_perc_cit_lb'] = min_diff(GEV_df['pred_perc_cit'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_perc_ind_lb'] = min_diff(GEV_df['pred_perc_ind'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    GEV_df['abs_diff_rev_2_lb'] = min_diff2d(GEV_df['REV_2_SCORE_lb'], GEV_df['REV_2_SCORE_ub'], GEV_df['REV_1_SCORE_lb'], GEV_df['REV_1_SCORE_ub'])
    
    ##%%
    MAD = GEV_df.median()[['abs_diff_ncs', 'abs_diff_njs', 'abs_diff_perc_cit', 'abs_diff_perc_ind', 'abs_diff_rev_2',
                           'abs_diff_ncs_lb', 'abs_diff_njs_lb', 'abs_diff_perc_cit_lb', 'abs_diff_perc_ind_lb', 'abs_diff_rev_2_lb',
                           'abs_diff_ncs_ub', 'abs_diff_njs_ub', 'abs_diff_perc_cit_ub', 'abs_diff_perc_ind_ub', 'abs_diff_rev_2_ub']]
    MAD['GEV_id'] = GEV_id
    MADs.append(MAD)
  ##%%
  MAD = pd.concat(MADs, axis=1).T.set_index('GEV_id')
  return MAD

#%% Run MAD calculations on individual level
MAD_ind = calc_MAD_ind(limited_df)

MAD_ind = MAD_ind.sort_index(axis=1)
MAD_ind = pd.merge(gev_names_df, MAD_ind, left_on='GEV_id', right_index=True)

#%%
sns.set_style('white')
sns.set_palette('Set1')
plt.figure(figsize=(9, 4))
xlabels = MAD['GEV']
x = np.arange(len(xlabels))
ycols = [('abs_diff_ncs', 'NCS'),
         ('abs_diff_njs', 'NJS'),
         ('abs_diff_perc_cit', 'Perc. Cit'),
         ('abs_diff_perc_ind', 'Perc. Journal'),
         ('abs_diff_rev_2', 'Review')]
gap = 0.01
group_gap = 0.3
width = (1 - group_gap - gap*(len(ycols) - 1))/len(ycols)
x = x - (len(ycols)*width + (len(ycols) - 1)*gap)/2 + width/2
bars = []
for idx, (ycol, ylabel) in enumerate(ycols):
  y = MAD[ycol]
  ybounds = MAD[[f'{ycol}_lb', f'{ycol}_ub']]
  b = plt.bar(x=x, height=y, color=colors[idx], alpha=0.5, width=width, label=ylabel)
  bars.append(b)
  plt.errorbar(x=x, y=y, fmt='none', capsize=2,
               yerr=np.abs(ybounds.subtract(y, axis='index').values.T),
               color=colors[idx])
  x += width + gap
plt.xticks(np.arange(len(xlabels)), xlabels, rotation='vertical')
#plt.xlabel('GEV')
plt.ylabel('MAD')
plt.legend(bars, list(zip(*ycols))[1],
           bbox_to_anchor=(0.5, 1.02), loc='center',
           ncol=5)
plt.grid(axis='y', ls=':')
sns.despine()
plt.savefig(os.path.join(results_dir, 'MAD.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(results_dir, 'MAD.png'), dpi=300, bbox_inches='tight')

#%%

sns.set_style('white')
sns.set_palette('Set1')
plt.figure(figsize=(9, 4))
xlabels = MAPD['GEV']
x = np.arange(len(xlabels))
ycols = [('abs_perc_diff_ncs', 'NCS'),
         ('abs_perc_diff_njs', 'NJS'),
         ('abs_perc_diff_perc_cit', 'Perc. Cit'),
         ('abs_perc_diff_perc_ind', 'Perc. Journal'),
         ('abs_perc_diff_rev_2', 'Review')]
gap = 0.01
group_gap = 0.3
width = (1 - group_gap - gap*(len(ycols) - 1))/len(ycols)
x = x - (len(ycols)*width + (len(ycols) - 1)*gap)/2 + width/2
bars = []
for idx, (ycol, ylabel) in enumerate(ycols):
  y = MAPD[ycol]
  ybounds = MAPD[[f'{ycol}_lb', f'{ycol}_ub']]
  b = plt.bar(x=x, height=y, color=colors[idx], alpha=0.5, width=width, label=ylabel)
  bars.append(b)
  plt.errorbar(x=x, y=y, fmt='none', capsize=2,
               yerr=np.abs(ybounds.subtract(y, axis='index').values.T),
               color=colors[idx])
  x += width + gap
plt.xticks(np.arange(len(xlabels)), xlabels, rotation='vertical')
#plt.xlabel('GEV')
plt.ylabel('MAPD')
plt.legend(bars, list(zip(*ycols))[1],
           bbox_to_anchor=(0.5, 1.02), loc='center',
           ncol=5)
plt.grid(axis='y', ls=':')
sns.despine()
plt.savefig(os.path.join(results_dir, 'MAPD.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(results_dir, 'MAPD.png'), dpi=300, bbox_inches='tight')

#%%
import statsmodels.formula.api as smf

df['abs_diff_rev_2'] = np.abs(df['REV_2_SCORE'] - df['REV_1_SCORE'])
mod = smf.ols('abs_diff_rev_2 ~ ncs', df)
res = mod.fit()
res.summary()
#%%
df['predict_missing_abs_diff_rev_2'] = res.predict(df)
df.groupby('missing_either')[['predict_missing_abs_diff_rev_2', 'abs_diff_rev_2']].median()
df.groupby('missing_either')[['predict_missing_abs_diff_rev_2', 'abs_diff_rev_2']].mean()

#%%
for GEV_id, GEV_df in df.groupby('GEV_id'):
  print(f'GEV: {GEV_id}')
  mod = smf.ols('abs_diff_rev_2 ~ ncs', GEV_df)
  res = mod.fit()
  print(res.summary())
  GEV_df['predict_missing_abs_diff_rev_2'] = res.predict(GEV_df)
  #df.groupby('missing_either')[['predict_missing_abs_diff_rev_2']].median()
  GEV_df.groupby('missing_either')[['predict_missing_abs_diff_rev_2']].mean()
