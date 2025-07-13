# imports
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap


# PLOT UTILS ------------------------------------------------------------------


# color for each prior_sd
colormap = {
    80: [0.5, 0, 0],
    40: [1, 0.2, 0],
    20: [1, 0.6, 0],
    10: [0.75, 0.75, 0]
}


# RAW DATA --------------------------------------------------------------------


# fetch data
url = 'https://github.com/steevelaquitaine/projInference/raw/gh-pages/data/csv/data01_direction4priors.csv'
try:
  RequestAPI = requests.get(url)
except requests.ConnectionError:
  print('Failed to download data. Please contact steeve.laquitaine@epfl.ch')
else:
  if RequestAPI.status_code != requests.codes.ok:
    print('Failed to download data. Please contact steeve.laquitaine@epfl.ch')
  else:
    with open('data01_direction4priors.csv', 'wb') as fid:
      fid.write(RequestAPI.content)

# read data
data = pd.read_csv('data01_direction4priors.csv')


# HELPERS: PROCESS DATA -------------------------------------------------------


def cart2pol(x:np.array, y:np.array):
  """
  Convert cartesian `(x,y)` to polar `(deg,mag)`.

  Args:
      `x`: scalar or array of x-coordinates
      `y`: scalar or array of y-coordinates

  Returns:
      Tuple of `(deg,mag)`, where `deg` is in the interval
      [0,360) counterclockwise from the positive x axis.
  """
  # compute
  deg = (np.degrees(np.arctan2(y,x)) + 360) % 360
  mag = np.hypot(x,y)
  return deg,mag


def pol2cart(deg: np.array, mag: np.array):
  """
  Convert polar `(deg,mag)` to cartesian `(x,y)`.

  Args:
      `deg`: scalar or array of angles in degrees [0,360)
      `mag`: scalar or array of magnitudes

  Returns:
      Tuple of `(x, y)` coordinates.
  """
  # compute
  rad = np.radians(deg)
  x = mag * np.cos(rad)
  y = mag * np.sin(rad)
  return x, y


def circdiff(angle:np.array, reference:np.array):
  """
  Compute the signed minimal circular distance from `angle` to `reference`.

  Args:
      `angle`: scalar or array of angles in degrees [0, 360)
      `reference`: scalar reference angle in degrees [0, 360)

  Returns:
      Signed circular distance in degrees (-180, 180].
  """
  # compute
  diff = ((angle - reference + 180) % 360) - 180
  diff[diff == -180] = 180
  return diff


def process_data(data:pd.DataFrame):
  """
  Take the orginal `data`, rename existing columns with clearer
  and shorter names, and add additional columns.

  Args:
    `data`: original dataframe

  Returns:
    Dataframe.
  """
  # columns to discard
  cols_remove = ['experiment_id', 'experiment_name', 'raw_response_time']
  # columns to rename
  cols_rename = {
    'subject_id'                  : 'subject_id',
    'session_id'                  : 'session_id',
    'run_id'                      : 'run_id',
    'trial_index'                 : 'trial_id',
    'trial_time'                  : 'trial_time',
    'prior_mean'                  : 'prior_mean',
    'prior_std'                   : 'prior_sd',
    'motion_direction'            : 'stim_deg',
    'motion_coherence'            : 'stim_coh',
    'response_arrow_start_angle'  : 'init_deg',
    'reaction_time'               : 'rt',
    'estimate_x'                  : 'resp_x',
    'estimate_y'                  : 'resp_y'
  }
  # final column order
  cols_final = [
    # experiment
    'subject_id', 'session_id', 'run_id', 'trial_id', 'trial_time',
    # stimuli
    'prior_mean', 'prior_sd', 'stim_deg', 'stim_deg_tm1', 'stim_deg_delta', 'stim_rel', 'stim_coh',
    # responses
    'init_deg', 'rt', 'resp_x', 'resp_y', 'resp_mag', 'resp_deg', 'resp_rel',
    # response errors
    'err', 'err_tm1', 'err_toprior', 'err_toprior_norm', 'err_awaytm1',
  ]
  # create deep copy
  df = data.copy()
  # drop or rename columns
  df.drop(cols_remove, axis=1, inplace=True)
  df.rename(columns=cols_rename, inplace=True)
  # add new columns...
  # ...stimuli
  df['stim_deg_tm1']   = df.groupby(['subject_id', 'run_id'])['stim_deg'].shift(1)
  df['stim_deg_delta'] = circdiff(df.stim_deg, df.stim_deg_tm1)
  df['stim_rel']       = circdiff(df.stim_deg, df.prior_mean)
  # ...responses
  df['resp_deg'], df['resp_mag'] = cart2pol(df.resp_x, df.resp_y)
  df['resp_rel'] = circdiff(df.resp_deg, df.prior_mean)
  # ...errors
  df['err'] = circdiff(df.resp_deg, df.stim_deg)
  df['err_tm1'] = df.groupby(['subject_id', 'run_id'])['err'].shift(1)
  df['err_toprior'] = np.where(df.stim_rel * df.err < 0, np.abs(df.err), -np.abs(df.err))
  df['err_toprior_norm'] = np.where(df.stim_rel != 0, df.err_toprior / np.abs(df.stim_rel), np.nan)
  df['err_awaytm1'] = np.where(df.stim_deg_delta * df.err < 0, np.abs(df.err), -np.abs(df.err))

  # reorder columns
  return df[cols_final]


#  HELPERS: PRINT INFO -----------------------------------------------------------------


def print_dfcols(df:pd.DataFrame, col_dict:dict[str, str], width:int=100):
  # check for missing and extra columns
  missing_cols = set(col_dict) - set(df.columns)
  extra_cols   = set(df.columns) - set(col_dict)
  if missing_cols:
    raise ValueError(f"Missing columns in df: {missing_cols}")
  if extra_cols:
    raise ValueError(f"Unexpected columns in df not in col_dict: {extra_cols}")
  # print formatted descriptions
  for col, desc in col_dict.items():
    print(f"{col}")
    for line in textwrap.wrap(desc, width=width):
      print(f"  {line}")


data_cols = {
  'subject_id'        : 'Integer identifying subject.',
  'session_id'        : 'Integer identifying session within a subject.',
  'run_id'            : 'Integer identifying the run within a session.',
  'trial_id'          : 'Integer identifying trial within run.',
  'trial_time'        : 'Start time of trial within run; first trial starts at 0.',
  'prior_mean'        : 'Prior mean in degrees, always 225.',
  'prior_sd'          : 'Prior standard deviation in degrees, one of 10,20,40,80.',
  'stim_deg'          : 'Stimulus orientation in degrees, one of 5,15,25,...355.',
  'stim_deg_tm1'      : 'Stimulus orientation in degrees from trial t-1.',
  'stim_deg_delta'    : 'Angular distance of `stim_deg` relative to `stim_deg_tm1`.',
  'stim_rel'          : 'Stimulus orientation relative to prior mean, in (-180,+180].',
  'stim_coh'          : 'Stimulus coherence, one of 6,12,24.',
  'init_deg'          : 'Initiation angle for response.',
  'rt'                : 'Reaction time.',
  'resp_x'            : 'Response x-coordinate (cartesian).',
  'resp_y'            : 'Response y-coordinate (cartesian).',
  'resp_deg'          : 'Response angle in degrees (polar), in [0,360).',
  'resp_mag'          : 'Response magnitude (polar).',
  'resp_rel'          : 'Response angle in degrees relative to prior mean, in (-180,+180].',
  'err'               : 'Response error in degrees.',
  'err_tm1'           : 'Response error in degrees from trial t-1.',
  'err_toprior'       : 'Same as `err` but sign is positive if towards `prior_mean`.',
  'err_toprior_norm'  : 'Same as `err_toprior` but normalized by `stim_rel`.',
  'err_awaytm1'       : 'Same as `err` but sign is positive if away from `stim_deg_tm1`.',
}


# PROCESS DATA, PRINT INFO  ---------------------------------------------------

df = process_data(data)
print_dfcols(df, data_cols)