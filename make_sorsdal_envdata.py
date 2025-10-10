"""
Make a file with the environmental variables of interest and tides from a tidal model, then shift all to UTC datetime and save as .csv file for future loading as an obspy stream. 
"""

import pandas as pd
from obspy.core import UTCDateTime, Stream, Trace
import glob
import os
import numpy as np
from scipy.signal import find_peaks

import pyTMD

# set the local time offset and location of the floating shelf.
offset = 7 * 60 * 60 #local time (Davis station) = UTCTime + 7:00 hours
lat = -68.70793
lon = 78.10155

#load in the environmental data from the weather station as a DataFrame
env_path = '/Users/jmagyar/Documents/SorsdalData/environmental'
tide_dir = '/Users/jmagyar/Documents/tides'
all_files = glob.glob(os.path.join(env_path, "*.csv"))

df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df = df.drop(['MAX_OBSERVATION_DATE','MIN_RAINFALL','AVG_RAINFALL','MAX_RAINFALL','MIN_WIND_SPEED',
                'MAX_WIND_SPEED','AVG_WIND_DIRECTION','MIN_RELATIVE_HUMIDITY','MAX_RELATIVE_HUMIDITY','MIN_AIR_TEMPERATURE',
                'MAX_AIR_TEMPERATURE','MIN_AIR_PRESSURE','MAX_AIR_PRESSURE'],axis=1)
df['AVG_RELATIVE_HUMIDITY'][df['AVG_RELATIVE_HUMIDITY'] > 100] = pd.NA

for index, row in df.iterrows():
    df['MIN_OBSERVATION_DATE_'][index] = pd.to_datetime(df['MIN_OBSERVATION_DATE_'][index],format='%Y-%m-%d %H:%M:%S')

#now resample at hourly intervals and give NaN values when there is missing hours
df = df.sort_values('MIN_OBSERVATION_DATE_',axis=0)
df = df.set_index('MIN_OBSERVATION_DATE_')
df = df.resample('1H').bfill(limit=1)

#make traces for the important climate variables
delta = 60 * 60 #seconds in an hour (sampling distance)
start = df.index[0]
starttime = UTCDateTime(start.year,start.month,start.day,start.hour,start.minute,start.second) 
#starttime -= offset


traces = []
for name in df.columns:
    tr = Trace(data=df[name].to_numpy())
    tr.stats.delta = delta
    tr.stats.starttime = starttime
    tr.stats.channel = name.split('_')[-1]
    traces.append(tr)

#combine the traces to make stream with all times
stream = Stream(traces=traces)

#get the times that it is sampled
times = stream[0].times().astype(np.float64)
#get tides from model at these times
epoch = stream[0].stats.starttime
epoch = (epoch.year,epoch.month,epoch.day,epoch.hour,epoch.minute,epoch.second)

model_times = np.mgrid[0:times[-1]:60] #1 minute intervals for the season 

tides = pyTMD.compute_tide_corrections(np.array([lon]),np.array([lat]),model_times,EPOCH=epoch,DIRECTORY=tide_dir,MODEL='CATS2008',TIME='UTC',TYPE='time series',METHOD='spline',EPSG='4326').flatten()
#add the tides as another trace in this stream

tide_tr = Trace(data=tides)
tide_tr.stats.delta = 60
tide_tr.stats.starttime = starttime
tide_tr.stats.channel = 'TIDES'
stream += Stream(traces=[tide_tr])

#now make a trace for the tidal phase
ht_ind = find_peaks(tides)[0]
lt_ind = find_peaks(-tides)[0]
ht_times = np.full_like(ht_ind,fill_value = UTCDateTime(*epoch), dtype=UTCDateTime) + model_times[ht_ind]
lt_times = np.full_like(lt_ind,fill_value = UTCDateTime(*epoch), dtype=UTCDateTime) + model_times[lt_ind]
lt_times_pos = lt_times
lt_times_neg = lt_times + 0.000001

ht_angle = np.resize([0],ht_times.size)
lt_angle_pos = np.resize([180],lt_times_pos.size)
lt_angle_neg = np.resize([-180],lt_times_neg.size)
angles = np.concatenate([ht_angle,lt_angle_pos,lt_angle_neg])
hl_times = np.concatenate([ht_times,lt_times_pos,lt_times_neg])

time_order = np.argsort(hl_times)
angles = (angles[time_order]).astype(np.float64)
peak_times = (hl_times[time_order] - UTCDateTime(*epoch)).astype(np.float64)

interp = np.interp(model_times,peak_times,angles) #interpolated into minutes

phase_tr = Trace(data=interp)
phase_tr.stats.delta = 60
phase_tr.stats.starttime = starttime
phase_tr.stats.channel = 'PHASE'
stream += Stream(traces=[phase_tr])


#now save this stream to file for later use
filename = os.path.join(env_path,'environmental_stream'+'.mseed')
save_stream = stream.split()
save_stream.write(filename, format='MSEED')
