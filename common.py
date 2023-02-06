import pandas as pd
import numpy as np

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

import matplotlib.pyplot as plt

def errorbar(x, y, ylow, yhigh=None, xlow=None, xhigh=None, **kwargs):
    yerr = None
    if yhigh is None:
        yerr = ylow
    else:
        yerr = [y - ylow, yhigh - y]
        
    xerr = None
    if xlow is not None and xhigh is None:
        xerr = xlow
    elif xlow is not None and xhigh is not None:
        xerr= [x - xlow, xhigh - x]
        
    return plt.errorbar(x, y, yerr=yerr, xerr=xerr, **kwargs)

def extract_variable(df, variable, axis, index_dtypes=None):
  """ Extracts a variable from a summary dataframe from stan, and vectorizes
  the indices. That is, if there is a variable x[1], x[2], etc.. in a stan
  dataframe, it is extracted by this function to 
  id1    x
  1      value
  2
  """
  
  import re
  
  # Find relevant indices (rows in this case)
  
  var_re = re.compile(f'{variable}\[[\w,]*\]')
  
  if axis == 'index' or axis == 0:
    
    variable_index = [idx for idx in df.index if var_re.match(idx) is not None]
    
    # Split the name of each matching index, i.e.
    # variable[1,2] will be split into 1, 2
    split_variable_indices = [idx[len(variable)+1:-1].split(',')
                                   for idx in variable_index]
    
    if not index_dtypes is None:
      n_indices = len(index_dtypes)
      for indices in split_variable_indices:
        for i in range(n_indices):
          indices[i] = index_dtypes[i](indices[i])
          
    variable_df = df.loc[variable_index,]
    
    variable_df.index = pd.MultiIndex.from_tuples(
      [(variable, ) + tuple(indices) for indices in split_variable_indices])
    
  elif axis == 'columns' or axis == 1:

    variable_columns = [col for col in df.columns if var_re.match(col) is not None]

    # Split the name of each matching index, i.e.
    # variable[1,2] will be split into 1, 2
    split_variable_columns = [idx[len(variable)+1:-1].split(',')
                                     for idx in variable_columns]
    
    # Ensure the type is correct for each index (if specified)
    if not index_dtypes is None:
      n_indices = len(index_dtypes)
      for indices in split_variable_columns:
        for i in range(n_indices):
          indices[i] = index_dtypes[i](indices[i])
          
    variable_df = df.loc[:,variable_columns]
      
    variable_df.columns = pd.MultiIndex.from_tuples(
      [(variable, ) + tuple(indices) for indices in split_variable_columns])

  else:
    raise ValueError(f"Unknown axis: {axis}")
    
  return variable_df

def percentile(n):
  """"Named percentile function to be used for aggregation of pandas DataFrames"""
  def percentile_(x):
    if not isinstance(x,pd.Series):
      raise ValueError('need Series argument')
    return np.percentile(x, n)
  percentile_.__name__ = f'percentile_{n}'
  return percentile_

from scipy import stats

#%% Define distributions
def normal(mu, sigma):
  return stats.norm(loc=mu, scale=sigma)

def exponential(inv_rate):
  return stats.expon(scale=inv_rate)

def lognormal(mu, sigma):
  return stats.lognorm(scale=np.exp(mu), s=sigma)

#%% Define functions for creating unique IDs

def unique_id(df, start_id=1):
    """ Create unique identifiers for each unique row in df."""
    return df.groupby(list(df.columns)).ngroup() + start_id

def nuniq(df):
    return df.drop_duplicates().shape[0]
