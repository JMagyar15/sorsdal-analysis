"""
Make a full broadband event catalogue for the entire Sorsdal deployment. Use the new Event and Chunk objects as code wrappers
"""

from obspy.core import UTCDateTime
from cryoquake import data_objects as do
from obspy.core.inventory import inventory
import os
import pandas as pd
from pathlib import Path



root = Path(__file__).parent.parent
w_path = root / "waveforms"
c_path = root / "catalogues"
s_path = root / "stations"
s_file = root / "stations" / "sorsdal_stations.xml" #there were some issues with the downloaded files so use this...

#download waveforms and stations if not already done, put in corresponding paths...
t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,15)
chunk = do.SeismicChunk(t1,t2)

#chunk.download_waveforms(str(w_path),str(s_path),'2A','BBS??','','HH?')

inv = inventory.read_inventory(s_file,level='response').select(station='BBS??')

avail_rows = []

for daychunk in chunk(24*60*60):
    daychunk.attach_waveforms(inv,w_path,buffer=60*60) #hour long buffer for filtering and STA/LTA
    daychunk.remove_response(pre_filt=[1,2,45,50],taper=False)
    daychunk.context('detect')
    daychunk.detect_events(c_path,trigger_type='multistalta',sta=0.2,lta=2.0,delta_sta=50,delta_lta=50,epsilon=2,thr_on=4,thr_off=3,thr_coincidence_sum=2,avg_wave_speed=1.5,thr_event_join=5.0) #needs to be at least 2 to avoid double event IDs - need better way to do this in future...

#! these seem like the right parameters, but need to get full data off macmini - local waveforms different for some reason?
"""
NOW COMPUTE THE DATA AVAILABILITY FOR THE ENTIRE SEASON AND SAVE TO CSV FOR PLOTTING.
"""
# avail_rows = []

# chunk.attach_waveforms(inv,w_path)
# split_stream = chunk.stream.slice(chunk.starttime,chunk.endtime).split()

# for tr in split_stream:
#     network, station, location, channel = tr.id.split('.')
#     row = {'Network':network,'Station':station,'Location':location,'Channel':channel,'Start':tr.stats.starttime,'End':tr.stats.endtime}
#     avail_rows.append(row)

# avail = pd.DataFrame(data=avail_rows)
# avail.to_csv('sorsdal_stream_availability.csv')