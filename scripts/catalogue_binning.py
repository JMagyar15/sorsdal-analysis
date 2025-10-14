from obspy.core import UTCDateTime
from cryoquake import data_objects as do
from obspy.core.inventory import inventory
import numpy as np
import os
from pathlib import Path

t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,15)

root = Path(__file__).parent.parent
w_path = root / "waveforms"
c_path = root / "catalogues"
spec_path = root / "spectrograms"
env_path = root / "environmental"
s_file = root / "stations" / "sorsdal_stations.xml"


inv = inventory.read_inventory(s_file,level='response')

chunk = do.SeismicChunk(t1,t2,time_offset=7) #local time is UTC + 07:00
event_cat = do.EventCatalogue(t1,t2,c_path) #get the event catalogue

"""BINNING PARAMETERS"""
daily_bins = 10*60 #10 minute bins for diurnal cycles
total_bins = 1*60*60 #hour long bins for entire season
tide_bins = 2.5 #width in degrees of bins
att_bin_num = 40 #number of bins to place attributes in


"""BIN CONSTRUCTION"""
total_bin_edges = np.mgrid[chunk.starttime:chunk.endtime:total_bins]
total_bin_centres = total_bin_edges + total_bins / 2
total_bin_edges_mt = np.array([time.matplotlib_date for time in total_bin_edges])

day_start = UTCDateTime(2018,1,1)
day_end = UTCDateTime(2018,1,2)

daily_bin_edges = np.mgrid[day_start:day_end:daily_bins]
daily_bin_centres = daily_bin_edges + daily_bins / 2
daily_bin_edges_mt = np.array([time.matplotlib_date for time in daily_bin_edges])

tide_bin_edges = np.mgrid[-180:180:tide_bins]
tide_bin_centres = tide_bin_edges + tide_bins / 2

"""ENVIRONMENTAL DATA"""
chunk.context('timeseries')
chunk.attach_environmental(env_path)
temp = chunk.env_stream.select(channel='TEM')[0]
tide = chunk.env_stream.select(channel='TID')[0]
phase = chunk.env_stream.select(channel='PHA')[0]


#GET EVENT TIMES IN REQUIRED FORMAT FOR BINNING (TIME OF YEAR, TIME OF DAY, TIDAL PHASE)
event_times_mt = [time.matplotlib_date for time in event_cat.event_times]
daily_event_times_mt = [UTCDateTime(2018,1,1,time.hour,time.minute,time.second).matplotlib_date for time in event_cat.event_times]
dt = (event_cat.event_times - tide.stats.starttime).astype(np.float64)
event_phase = np.interp(dt,phase.times(),phase.data)

"""
SPECTROGRAM BINNING
"""


for code in inv.get_contents()['channels']:
    network, station, location, channel = code.split('.')


    chunk.context('spectral')
    spec_dict = chunk.load_periodograms(inv.select(network=network,station=station,channel=channel),spec_path)

    print('Binning spectrograms for',code)
    t, f, spec = spec_dict[code] #separate into the arrays

    #interpolate the environmental variables onto these PSD times
    dt = (t - chunk.starttime).astype(np.float64)

    interp_phase = np.interp(dt,phase.times(),phase.data)
    interp_tides = np.interp(dt,tide.times(),tide.data)
    interp_temp = np.interp(dt,temp.times(),temp.data)
    
    #firstly do the full season binning.
    t_mt = np.array([time.matplotlib_date for time in t])
    bin_ind = np.digitize(t_mt,total_bin_edges_mt) - 1

    medians = []
    ave_temp = []
    ave_tide = []
    for i, centre in enumerate(total_bin_centres):
        psds = spec[:,bin_ind==i]
        medians.append(np.nanmedian(psds,axis=1))
        ave_temp.append(np.nanmean(interp_temp[bin_ind==i]))
        ave_tide.append(np.nanmean(interp_tides[bin_ind==i]))

    binned_spec = np.stack(medians,axis=1)
    binned_temp = np.array(ave_temp)
    binned_tide = np.array(ave_tide)

    binned_events = np.bincount(np.digitize(event_times_mt,total_bin_edges_mt)-1,minlength=total_bin_edges_mt.size)

    #now save this binned spectrogram, along with the times, frequencies, and environmental drivers
    filename = os.path.join(spec_path,'full_spectrogram__'+ code + '__' + chunk.str_name + '.npz')
    np.savez(filename,spec=binned_spec,t=total_bin_centres,f=f,temp=binned_temp,tide=binned_tide,events=binned_events)


    #now do the dirunal binning
    t_mt = np.array([UTCDateTime(2018,1,1,time.hour,time.minute,time.second).matplotlib_date for time in t])
    bin_ind = np.digitize(t_mt,daily_bin_edges_mt) - 1

    medians = []
    ave_temp = []
    for i, centre in enumerate(daily_bin_centres):
        psds = spec[:,bin_ind==i]
        medians.append(np.nanmedian(psds,axis=1))        
        ave_temp.append(np.nanmean(interp_temp[bin_ind == i]))
    
    binned_spec = np.stack(medians,axis=1)
    binned_temp = np.array(ave_temp)

    binned_events = np.bincount(np.digitize(daily_event_times_mt,daily_bin_edges_mt)-1,minlength=daily_bin_edges_mt.size)
    
    filename = os.path.join(spec_path,'diurnal_spectrogram__'+ code + '__' + chunk.str_name + '.npz')
    np.savez(filename,spec=binned_spec,t=daily_bin_centres,f=f,temp=binned_temp,events=binned_events)

    #finally do the tidal binning
    bin_ind = np.digitize(interp_phase,tide_bin_edges) - 1

    medians = []
    ave_tide = []
    for i, centre in enumerate(tide_bin_centres):
        psds = spec[:,bin_ind==i]
        medians.append(np.nanmedian(psds,axis=1))
        ave_tide.append(np.nanmean(interp_tides[bin_ind==i]))

    binned_spec = np.stack(medians,axis=1)
    binned_tide = np.array(ave_tide)

    binned_events = np.bincount(np.digitize(event_phase,tide_bin_edges)-1,minlength=tide_bin_edges.size)


    filename = os.path.join(spec_path,'tide_phase_spectrogram__'+ code + '__' + chunk.str_name + '.npz')
    np.savez(filename,spec=binned_spec,phase=tide_bin_centres,f=f,tide=binned_tide,events=binned_events)