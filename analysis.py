import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
df = pd.read_excel('../data/db_VQR10all20171027.xlsx')
df = df.set_index(['INSTITUTION_ID', 'GEV']);
df['REV_1_SCORE'] = df[['REV_1_ORIGINALITY', 'REV_1_RIGOR', 'REV_1_IMPACT']].sum(axis=1)
df['REV_2_SCORE'] = df[['REV_2_ORIGINALITY', 'REV_2_RIGOR', 'REV_2_IMPACT']].sum(axis=1)
df['REV_SCORE'] = df[['REV_1_ORIGINALITY', 'REV_1_RIGOR', 'REV_1_IMPACT', 
                      'REV_2_ORIGINALITY', 'REV_2_RIGOR', 'REV_2_IMPACT']].sum(axis=1)
#%%
rev_col = [c for c in df.columns if 'REV' in c]
c = df.corr()
#%% Calculate outcome results on institutional level
g = df.groupby(level=[0, 1])
cols = ['PEER_REVIEW_SCORE', 'PERCENTILE_CITATIONS', 'PERCENTILE_INDICATOR_VALUE']
inst_df = g[cols].mean()
inst_df['PEER_REVIEW_SCORE'] /= 6;
inst_df['PERCENTILE_CITATIONS'] /= 10;
inst_df['PERCENTILE_INDICATOR_VALUE'] /= 10;

def class_from_score(x):
    if 1 <= x < 2.5:
        return 'Limited'
    elif 2.5 <= x <= 5.2:
        return 'Acceptable'
    elif 5.2 < x <= 7.2:
        return 'Fair'
    elif 7.2 < x < 9:
        return 'Good'
    elif 9 <= x <= 10:
        return 'Excellent'
    else:
        return 'N/A'

cols_class = [c + '_CLASS' for c in cols]
inst_df[cols_class] = inst_df.applymap(class_from_score)
#%% Put institutional correlations back in
cols = ['INSTITUTION_' + c for c in inst_df.columns]
df[cols] = inst_df;
#%%
#%% Correlations on individual level, per GEV.
cols = ['PEER_REVIEW_SCORE', 'REV_1_CLASS_SCORE', 'REV_2_CLASS_SCORE', 
        'REV_1_SCORE', 'REV_2_SCORE',
        'INDICATOR_VALUE', 'PERCENTILE_INDICATOR_VALUE',
       'CITATIONS_NUMBER', 'PERCENTILE_CITATIONS', ]
c = df.groupby(level=['GEV'])[cols].corr()
#%% Correlations on institutional level, per GEV and used DB.
c = df.groupby(['GEV', 'USED_DB'])[cols].corr()
c = c.unstack('USED_DB')

x = c.loc[(slice(None), 'REV_1_SCORE'), 
          ['REV_1_SCORE', 'REV_2_SCORE', 'PERCENTILE_CITATIONS', 'PERCENTILE_INDICATOR_VALUE']]
#%% Correlations on institutional level, per GEV.
inst_df = df.groupby(['INSTITUTION_ID', 'GEV']).mean()
c = inst_df.groupby(level=['GEV'])[cols].corr()

x = c.loc[(slice(None), ['REV_1_SCORE', 'REV_2_SCORE']), 
          ['REV_1_SCORE', 'PERCENTILE_CITATIONS', 'PERCENTILE_INDICATOR_VALUE']]\
     .unstack(level=1)
#%% Correlations on institutional level, per GEV and used DB.
c = df.groupby(['INSTITUTION_ID', 'GEV', 'USED_DB']).mean().groupby(level=['GEV', 'USED_DB'])[cols].corr()
c = c.unstack('USED_DB')

x = c.loc[(slice(None), 'REV_1_SCORE'), 
          ['REV_1_SCORE', 'REV_2_SCORE', 'PERCENTILE_CITATIONS', 'PERCENTILE_INDICATOR_VALUE']]
#%% Correlations for same country or not
df['REV_SAME_COUNTRY'] = df['REV_1_AFFILIATION'] == df['REV_2_AFFILIATION']
c = df.groupby('REV_SAME_COUNTRY')[cols].corr()
#%%
c = df.groupby(['REV_1_AFFILIATION', 'REV_2_AFFILIATION'])[cols].corr()
#%% Correlations for different journal qualities
c = df.groupby('VQR_PERCENTILE_INDICATOR_VALUE')[cols].corr()
#%% Correlations for different institutional qualities
#c = df.groupby('INSTITUTION_PERCENTILE_CITATIONS_CLASS')[cols].corr()
c = df.groupby('INSTITUTION_PEER_REVIEW_SCORE_CLASS')[cols].corr()
#%% Plot Scopus vs Web of Science citations and percentiles
for db, db_df in df.query('GEV == 3').groupby('USED_DB'):
#    plt.plot(db_df['INDICATOR_VALUE'], db_df['PERCENTILE_INDICATOR_VALUE'], 'o', label=db, alpha=0.2)
    plt.plot(db_df['CITATIONS_NUMBER'], db_df['PERCENTILE_CITATIONS'], 'o', label=db, alpha=0.2)
plt.xscale('log')
plt.legend(loc='best')