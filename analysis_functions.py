import pandas as pd

def extract_variable(df, variable, axis):
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
    
    variable_df = df.loc[variable_index,]
    
    variable_df.index = pd.MultiIndex.from_tuples(
      [(variable, ) + tuple(indices) for indices in split_variable_indices])
  
  elif axis == 'columns' or axis == 1:

    variable_columns = [col for col in df.columns if var_re.match(col) is not None]

    # Split the name of each matching index, i.e.
    # variable[1,2] will be split into 1, 2
    split_variable_columns= [idx[len(variable)+1:-1].split(',')
                                   for idx in variable_columns]
    
    variable_df = df.loc[:,variable_columns]
      
    variable_df.columns = pd.MultiIndex.from_tuples(
      [(variable, ) + tuple(indices) for indices in split_variable_columns])
  else:
    raise ValueError(f"Unknown axis: {axis}")
    
  return variable_df