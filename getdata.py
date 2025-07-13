# imports
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


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
  return ((angle - reference + 180) % 360) - 180


def process_data(data:pd.DataFrame):
  """
  Take the orginal `data`, rename existing columns with clearer
  and shorter names, and add additional columns.

  Args:
    `data`: original dataframe

  Returns:
    Dataframe with the following columns:
      - `subject_id` (int identifying subject)
      - `session_id` (int identifying session within a subject)
      - `run_id` (int identifying the run or block with a session)
      - `trial_id` (int identifying trial within run)
      - `trial_time` (start time of trial, first trial of run starts at 0)
      - `prior_mean` (prior mean in deg, always 225)
      - `prior_sd` (prior sd in deg, one of 10,20,40,80)
      - `stim_deg` (stimulus orientation in deg, one of 5,15,25,...355)
      - `stim_rel` (stimulus orientation relative to prior mean, from -180 to 180)
      - `stim_coh` (stimulus coherence, one of 6,12,24)
      - `init_deg` (initiation angle for response)
      - `rt` (reaction time)
      - `resp_x, resp_y`(cartesian response coords: x and y)
      - `resp_deg, resp_mag` (polar response coords: degrees and magnitude)
      - `resp_rel` (response degrees relative to prior mean, from -180 to 180)
      - `err` (response error in degrees)
      - `err_prev` (previous trial response error within a given run)
      - `err_prior` (same as `err` but sign is positive if in direction of prior mean)
      - `err_priornorm` (same as `err_prior` but normalized to distance of stimulus to prior mean)
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
    'subject_id', 'session_id', 'run_id', 'trial_id', 'trial_time',
    'prior_mean', 'prior_sd', 'stim_deg', 'stim_rel', 'stim_coh',
    'init_deg', 'rt', 'resp_x', 'resp_y', 'resp_deg', 'resp_rel',
    'err', 'err_prev', 'err_prior', 'err_priornorm'
  ]
  # create deep copy
  df = data.copy()
  # drop or rename columns
  df.drop(cols_remove, axis=1, inplace=True)
  df.rename(columns=cols_rename, inplace=True)
  # add new columns
  df['stim_rel'] = circdiff(df.stim_deg, df.prior_mean)
  df['resp_deg'], df['resp_mag'] = cart2pol(df.resp_x, df.resp_y)
  df['resp_rel'] = circdiff(df.resp_deg, df.prior_mean)
  df['err'] = circdiff(df.resp_deg, df.stim_deg)
  df['err_prev'] = df.groupby(['subject_id', 'run_id'])['err'].shift(1)
  df['err_prior'] = np.where(df.stim_rel * df.err < 0, np.abs(df.err), -np.abs(df.err))
  df['err_priornorm'] = np.where(df.stim_rel != 0, df.err_prior / np.abs(df.stim_rel), np.nan)

  # reorder columns
  return df[cols_final]


# PROCESS DATA ----------------------------------------------------------------

df = process_data(data)
# print info
df.head(10)