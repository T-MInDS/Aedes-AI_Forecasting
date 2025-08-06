import numpy as np
import pandas as pd
import os, glob, json
from sklearn.metrics.pairwise import euclidean_distances
import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

# Opening config file
f = open('../fpaths_config.json')
paths = json.load(f)

weather_path = paths["weather_data"]
nn_preds_path = paths["nn_abundance_predictions"]
raw_mols_path = paths["raw_mols"]

def collect_locs_dists():
    return ['Arboleda', 'La_Margarita', 'Villodas', 'San_Juan'], ['negbin', 'poisson']

def load_csv(file):
  if os.path.exists(file):
    data = pd.read_csv(file)
  else:
    print('File does not exist')
  return data

def save_weather_mols():
  san_juan = pd.read_pickle('{}/San_Juan_daily.pd'.format(weather_path))
  san_juan.to_csv('{}/San_Juan_MoLS.csv'.format(raw_mols_path), index=False)

  ceiba = pd.read_pickle('{}/Ceiba_daily.pd'.format(weather_path))
  ceiba.to_csv('{}/Ceiba_MoLS.csv'.format(raw_mols_path), index=False)

  west_sj = pd.read_pickle('{}/West_SJ_daily.pd'.format(weather_path))
  west_sj.to_csv('{}/West_SJ_MoLS.csv'.format(raw_mols_path), index=False)
  return

def format_lcd(data_path, loc):
  # First format weather data
  data = pd.read_csv(data_path)
  data['Datetime'] = pd.to_datetime(data['DATE'])
  data.drop(columns=['STATION', 'DATE', 'REPORT_TYPE', 'SOURCE'], inplace=True)
  #Replace trace amounts of precipitation
  data[['DailyPrecipitation', 'HourlyPrecipitation']] = data[['DailyPrecipitation', 'HourlyPrecipitation']].replace('T', 0)
  #Ensure columns are numeric
  cols = ['DailyAverageDryBulbTemperature', 'DailyAverageRelativeHumidity', 'DailyPrecipitation', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyRelativeHumidity']
  for col in cols:
    data[col] = pd.to_numeric(data[col]).astype(float)
  #Convert hourly data to daily
  data_daily = data.groupby(data['Datetime'].dt.date).agg({
    'DailyAverageDryBulbTemperature': 'max',
    'DailyAverageRelativeHumidity': 'max',
    'DailyPrecipitation': 'max',
    'HourlyDryBulbTemperature': 'mean',
    'HourlyPrecipitation': 'sum',
    'HourlyRelativeHumidity': 'mean'
  })
  data_daily.reset_index(inplace=True)

  #Replace NaN in daily data with hourly average
  data_daily['DailyAverageDryBulbTemperature'] = data_daily['DailyAverageDryBulbTemperature'].fillna(data_daily['HourlyDryBulbTemperature'])
  data_daily['DailyAverageRelativeHumidity'] = data_daily['DailyAverageRelativeHumidity'].fillna(data_daily['HourlyRelativeHumidity'])
  data_daily['DailyPrecipitation'] = data_daily['DailyPrecipitation'].fillna(data_daily['HourlyPrecipitation'])

  #Split datetime into year month day columns
  data_daily['Datetime'] = pd.to_datetime(data_daily['Datetime'])
  data_daily['Year'] = data_daily['Datetime'].dt.year
  data_daily['Month'] = data_daily['Datetime'].dt.month
  data_daily['Day'] = data_daily['Datetime'].dt.day

  #Rename columns
  data_daily.rename(columns={'DailyAverageDryBulbTemperature': 'Avg_Temp', 'DailyAverageRelativeHumidity': 'Humidity', 'DailyPrecipitation': 'Precip_in'}, inplace=True)

  #Convert Fahrenheit to Celsius
  data_daily['Avg_Temp'] = 5/9*(data_daily['Avg_Temp'] - 32.0)

  #Get correct order of columns
  data_daily['Location'] = loc
  data_daily['Ref'] = 0
  data_daily = data_daily[['Location', 'Year', 'Month', 'Day', 'Avg_Temp', 'Precip_in', 'Humidity', 'Ref']]
  data_daily['Datetime'] = pd.to_datetime(data_daily[['Year', 'Month', 'Day']])
  data_cols = ['Avg_Temp', 'Precip_in', 'Humidity']


  data_daily = impute_vals(data_daily, data_cols)
  data_daily = interpolate_outliers(data_daily, data_cols, [], noise_std=0.05, threshold=100)
  

  data_daily.to_csv('../data/Weather/{}.csv'.format(loc), index=False)

  data_daily['Precip'] = 25.4 * data_daily['Precip_in']
  data_daily = data_daily[['Location', 'Year', 'Month', 'Day', 'Avg_Temp', 'Precip', 'Humidity', 'Ref']]
  data_daily.to_pickle('../data/Weather/{}.pd'.format(loc))
  return data_daily


def ceiba_missing_dates():
  missing_vals = [('2007-12-15', '2010-09-07'), 
                  ('2004-03-26', '2005-10-18')]
  return missing_vals

def collect_ceiba():
  #First merge Ceiba weather sources
  #Asos
  asos = pd.read_csv('{}/ceiba/tjnr_asos.csv'.format(weather_path))
  asos['Datetime'] = pd.to_datetime(asos['day'])
  asos.set_index('Datetime', inplace=True)

  #reindex to include all days
  full_idx = pd.date_range(start = asos.index.min(), end = asos.index.max(), freq='D')
  asos = asos.reindex(full_idx)

  asos['Humidity'] = (asos['min_rh'] + asos['max_rh']) / 2
  asos['Humidity'] = asos['Humidity'].fillna(asos['avg_rh'])
  asos['avg_temp_f'] = (asos['max_temp_f'] + asos['min_temp_f']) / 2


  #Coop
  coop_crlp = pd.read_csv('{}/ceiba/crlp4_coop.csv'.format(weather_path))
  coop_crlp['Datetime'] = pd.to_datetime(coop_crlp['day'])
  coop_crlp.set_index('Datetime', inplace=True)
  coop_crlp = coop_crlp.reindex(full_idx)

  coop_crlp.drop(columns=['day', 'min_rh', 'avg_rh', 'max_rh'], inplace=True)
  coop_crlp['avg_temp_f'] = (coop_crlp['min_temp_f'] + coop_crlp['max_temp_f']) / 2

  coop_ngbp = pd.read_csv('{}/ceiba/ngbp4_coop.csv'.format(weather_path))
  coop_ngbp['Datetime'] = pd.to_datetime(coop_ngbp['day'])
  coop_ngbp.set_index('Datetime', inplace=True)
  coop_ngbp = coop_ngbp.reindex(full_idx)

  coop_ngbp['precip_in'] = coop_ngbp['precip_in'].fillna(coop_crlp['precip_in'])
  coop_ngbp.drop(columns=['day', 'max_temp_f', 'min_temp_f', 'min_rh', 'avg_rh', 'max_rh'], inplace=True)
  coop_crlp.drop(columns=['station', 'precip_in'], inplace=True)

  full = pd.merge(coop_ngbp, coop_crlp, left_index=True, right_index=True, how='left')
  cols = ['precip_in', 'max_temp_f', 'min_temp_f', 'avg_temp_f']

  for col in cols:
      full[col] = full[col].fillna(asos[col])

  #Now merge asos and coop
  full.drop(columns=['station', 'max_temp_f', 'min_temp_f'], inplace=True)
  asos.drop(columns=['station', 'day', 'max_temp_f', 'min_temp_f', 'precip_in', 'avg_temp_f', 'avg_rh', 'min_rh', 'max_rh'], inplace=True)

  full = pd.merge(full, asos, left_index=True, right_index=True, how='left')
  full.rename(columns={'precip_in': 'Precip_in', 'avg_temp_f': 'Avg_Temp'}, inplace=True)
  full.reset_index(inplace=True)
  full.rename(columns={'index': 'Datetime'}, inplace=True)

  full.to_csv('{}/DailySummaries_Ceiba.csv'.format(weather_path), index=False)
  return



def impute_vals(data, data_cols, exclude_ranges = [],  noise_std_frac = 0.05):
  
  filled = data.copy(deep=True)

  for idx, row in filled.iterrows():
    row_date = row.Datetime
    # Skip rows within any of the excluded date ranges
    if any(start <= row_date <= end for start, end in [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in exclude_ranges]):
        continue

    #Num missing
    num_known = row[data_cols].notnull().sum()

    #Skip rows with fewer than 2 known values
    if num_known < 2:
        continue

    #Proceed if there's at least one missing value
    if row[data_cols].isnull().any():        
      #columns with missing data in this row
      missing_cols = row[data_cols][row[data_cols].isnull()].index.tolist()
      #columns with valid data
      valid_cols = row[data_cols][row[data_cols].notnull()].index.tolist()

      #Filter rows that have values in all of the valid_cols
      candidates = data.dropna()[data_cols]
      
      if not candidates.empty:
        #Compute distances between the current row and candidate rows
        row_vals = row[valid_cols].values.reshape(1, -1)
        candidate_vals = candidates[valid_cols].values
            
        distances = euclidean_distances(row_vals, candidate_vals).flatten()
        best_match_idx = distances.argmin()
            
        best_match_row = candidates.iloc[best_match_idx]
            
        #Now fill in missing values
        for col in missing_cols:
          value = best_match_row[col]
          noise = np.random.normal(loc = 0, scale = noise_std_frac*abs(value))
          filled.at[idx, col] = value + noise
  return filled

def interpolate_outliers(data, data_cols, exclude_ranges =  [], noise_std=0.05, threshold = 3):
  data['Datetime'] = pd.to_datetime(data['Datetime'])
  interpolated = data.copy()
    
  # Create a mask for excluded date ranges
  exclusion_ranges = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in exclude_ranges]

  #Create a mask of rows in any exclusion range
  mask = pd.Series(False, index=data.index)
  for start,end in exclusion_ranges:
    mask = mask | data.Datetime.between(start,end)

  #Excluded ranges: 2905-3902, 1546-2117
  for col in data_cols:
    #Create a temporary series for interpolation
    temp_series = data[col].copy()
    
    #Find nan values outside exclusion range    
    nan_indices = temp_series[~mask][temp_series[~mask].isna()].index      
    
    # Perform interpolation
    interpolated_series = temp_series[~mask].interpolate(method='linear')
    
    # Add noise only to the interpolated values
    noise = np.random.normal(loc=0.0, scale=noise_std*data[col].mean(), size=len(nan_indices))
    interpolated_series[nan_indices] += noise
    interpolated_series[interpolated_series < 0] = 0

    # Assign back to the dataframe
    interpolated[col] = interpolated_series
    #missing values
    #impute outliers
    #outliers = (np.abs(interpolated[col] - interpolated[col].mean()) > threshold * interpolated[col].std())
  
    #for i in range(0,len(interpolated)):
    #  if outliers.iloc[i]:
    #    prev_val = interpolated[col].iloc[i-1]
    #    noise = np.random.normal(0, 0.1*interpolated[col].mean())
    #    interpolated.loc[i, col] = prev_val + noise

  return interpolated


def cleanDailySummaries(data, loc):

  data = data[['Datetime', 'Precip_in', 'Humidity', 'Avg_Temp']]
  data['Datetime'] = pd.to_datetime(data['Datetime'])
  
  # Columns to use for similarity (exclude 'day')
  data_cols = ['Precip_in', 'Humidity', 'Avg_Temp']
  if loc=='Ceiba':
    missing_vals = ceiba_missing_dates()
  else:
    missing_vals = []
  filled = impute_vals(data, data_cols, missing_vals)
  filled = interpolate_outliers(filled, data_cols, missing_vals, threshold=3)      
        
  filled['Location'] = loc
  filled['Year'] = filled.Datetime.dt.year
  filled['Month'] = filled.Datetime.dt.month
  filled['Day'] = filled.Datetime.dt.day
  #filled.rename(columns={'avg_rh': 'Humidity', 'precip_in': 'Precip_in', 'avg_temp_f': 'Avg_Temp'}, inplace=True)
  filled['Ref'] = 0

  filled = filled[['Location', 'Datetime', 'Year', 'Month', 'Day', 'Avg_Temp', 'Precip_in', 'Humidity', 'Ref']]
  filled['Avg_Temp'] = 5/9 * (filled['Avg_Temp'] - 32.0)

  filled.to_csv('../data/weather/{}.csv'.format(loc), index=False)
  return filled


def partition_week(th=0.05):
  # Returns 7 numbers that add up to 1, of minimum value set by th (before
  # normalization)
  # Set th to a large value (e.g. 0.5) for identical numbers in pt
  rng = np.random.default_rng()
  pt=np.sort(rng.random(10)); # Define 9 random numbers between 0 and 1
  pt=pt[1:]-pt[:-1]; # Take length of each interval
  pt=(pt[1:-1]+pt[:-2]+pt[2:])/3 # Three-point moving average
  pt[pt<th]=0; # Low-precipitation threshold
  if np.sum(pt)==0:
    pt=np.ones(7)/7
  else:
    pt=pt/np.sum(pt); # Normalize
  return pt

def create_AedesAI_input_dataframe(indata,th=0.05):
  indata.Datetime = pd.to_datetime(indata.Datetime)
  # Find range of weeks in input dataframe
  wk=list(set(indata.Week)); wmin=np.min(wk); wmax=np.max(wk)
  # For each week, create temperature and RH data based on mean and standard
  # deviation; distribute rainfall over the week, using partition_week(th)
  # Also create an estimate of the number of females based on reported mean
  # and standard deviation
  T=[]; RH=[]; PT=[]; th=0.05
  rng = np.random.default_rng()
  
  for idx in range(wmax-wmin):
    # Temperature
    mu=indata.TempWeek_C[idx]; sg=indata.TempWeekSD[idx]
    T=np.append(T,rng.normal(mu, sg, 7))
    # Relative humidity
    mu=indata.RHWeek_pct[idx]; sg=indata.RHWeekSD[idx]
    RH=np.append(RH,rng.normal(mu, sg, 7))
    # Precipitation
    PT=np.append(PT,indata.RainWeek_mm[idx]/10*partition_week(th))
  # Create dataframe with daily data
  Ddata=pd.DataFrame(
      {
      'Avg_Temp': pd.Series(T),
      'Humidity': pd.Series(RH),
      'Precip': pd.Series(PT),
      },
      index=np.arange(len(T))
  )
  # Assign dates to all entries, the first one of which is 3 days (Monday)
  # before the Wednesday of the first week
  dtes=indata.Datetime-pd.DateOffset(days=3)
  Ddata['Date']=pd.Series(pd.date_range(dtes[0], periods=len(T)))
  Ddata['Year'], Ddata['Month'] = Ddata['Date'].dt.year, Ddata['Date'].dt.month
  Ddata['Day'] = Ddata['Date'].dt.day
  # Add location column
  Ddata['Location'] = indata.Site[0]
  # Reorder the columns
  Ddata=Ddata[['Location', 'Year', 'Month', 'Day', 'Avg_Temp', 'Precip','Humidity']]
  return Ddata, Ddata.columns

def add_missing_values(indata):
  # Find week range
  wk=list(set(indata.Week)); wmin=np.min(wk); wmax=np.max(wk)
  # Find missing weeks - concatenate list of all weeks with weeks in indata dataframe
  w2=indata; w2.index=w2.Week
  wks=pd.DataFrame({'All_Week': np.arange(wmin,wmax)},
                 index=np.arange(wmax-wmin)+np.min(indata.index))
  tt=pd.concat([wks,w2],axis=1)
  missing_weeks=tt.loc[lambda tt: pd.isna(tt.Site), :].index
  # Bracket range of missing weeks
  tmp=missing_weeks[0]+np.arange(len(missing_weeks)+2)-1

  ist=tmp[0]
  iend=tmp[-1]
  # Update week number
  tt.loc[missing_weeks,'Week']=tt.loc[missing_weeks,'All_Week']
  # Update site name
  tt.loc[missing_weeks,'Site']=tt.loc[ist,'Site']
  
  # Update numerical values
  for id in tt.columns[5:]:     # Starts at 4 because added 'All_Week' column
    tmp=tt.loc[ist,id]+np.arange(len(missing_weeks)+2)/(len(missing_weeks)+1)*(tt.loc[iend,id]-tt.loc[ist,id])
    tt.loc[missing_weeks,id]=tmp[1:-1]
  # Return dataframe with original columns
  tt=tt[indata.columns]
  tt.Week=tt.Week.astype(int)
  # Reindex the rows starting from 0
  tt.index=np.arange(tt.shape[0])
  return tt

def replace_nan_values(indata,cst):
  tt=indata
  nan_week=tt.loc[lambda tt: pd.isna(tt.RHWeekSD), :].index
  # Bracket range of missing weeks
  tmp=nan_week[0]+np.arange(len(nan_week)+2)-1
  ist=tmp[0]; iend=tmp[-1]
  # Update numerical values
  for id in tt.columns[cst:]:    # Starts at column cst
    tmp=tt.loc[ist,id]+np.arange(len(nan_week)+2)/(len(nan_week)+1)*(tt.loc[iend,id]-tt.loc[ist,id])
    tt.loc[nan_week,id]=tmp[1:-1]
  tt.index=np.arange(tt.shape[0])
  return tt