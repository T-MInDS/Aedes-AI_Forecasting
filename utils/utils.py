import numpy as np
import pandas as pd
import os, glob

def merge_for_training(fil_path, dist):
  samples = pd.read_csv('{}/Arboleda_{}.csv'.format(fil_path, dist))
  samples['Location'] = 'Arboleda'

  to_merge = pd.read_csv('{}/Villodas_{}.csv'.format(fil_path, dist))
  to_merge['Location'] = 'Villodas'
  samples = pd.concat([samples, to_merge], axis=0)

  to_merge = pd.read_csv('{}/La_Margarita_{}.csv'.format(fil_path, dist))
  to_merge['Location'] = 'La_Margarita'
  samples = pd.concat([samples, to_merge], axis=0)

  to_merge = pd.read_csv('{}/San_Juan_{}.csv'.format(fil_path, dist))
  to_merge['Location'] = 'San_Juan'
  samples = pd.concat([samples, to_merge], axis=0)
  
  samples.Datetime = pd.to_datetime(samples.Datetime)
  samples['Year'] = samples.Datetime.dt.year
  samples['Month'] = samples.Datetime.dt.month
  samples['Day'] = samples.Datetime.dt.day

    
  weather = pd.read_pickle('../data/Weather/Arboleda_daily.pd')
  to_merge = pd.read_pickle('../data/Weather/La_Margarita_daily.pd')
  weather = pd.concat([weather, to_merge], axis=0)

  to_merge = pd.read_pickle('../data/Weather/Villodas_daily.pd')
  weather = pd.concat([weather, to_merge], axis=0)

  to_merge = pd.read_pickle('../data/Weather/San_Juan_daily.pd')
  weather = pd.concat([weather, to_merge], axis=0)

  merged = weather.merge(samples, on=['Location', 'Year', 'Month', 'Day'], how='left')
  merged.drop(columns=['Ref', 'Neural Network'], inplace=True)
  merged = merged[['Location', 'Year', 'Month', 'Day', 'Avg_Temp', 'Precip', 'Humidity', 'weekly_mean', 'weekly_var']]
  
  merged.to_csv('../utils/model_files/finetune_training_traps.csv', index=False)
  return merged

def format_synthetic_for_training(data, scaler):
  X_train = [], y_train = [], X_val = [], y_val = [], X_test = [], y_test = []
  for i in range(0,len(data)):
    sample = data.iloc[i:i+90]
    if not pd.isna(sample.iloc[-1]['weekly_mean']):
      scaled = scaler.transform(sample)
      if sample.Year<2015:
        
        X_train.append(scaler.transform(sample[['Avg_Temp', 'Precip', 'Humidity']].values))

  print(data)
  asdf



def load_csv(file):
  if os.path.exists(file):
    data = pd.read_csv(file)
  else:
    print('File does not exist')
  return data


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