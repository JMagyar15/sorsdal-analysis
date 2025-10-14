
from obspy.core import UTCDateTime
from cryoquake import data_objects as do
from obspy.core.inventory import inventory
import pandas as pd
from pathlib import Path


t1 = UTCDateTime(2017,12,31)
t2 = UTCDateTime(2018,2,16) 

root = Path(__file__).parent.parent
w_path = root / "waveforms"
s_file = root / "stations" / "sorsdal_stations.xml"

inv = inventory.read_inventory(s_file,level='response')
chunk = do.SeismicChunk(t1,t2,time_offset=7)

avail_rows = []

chunk.attach_waveforms(inv,w_path)
split_stream = chunk.stream.slice(chunk.starttime,chunk.endtime).split()

for tr in split_stream:
    network, station, location, channel = tr.id.split('.')
    row = {'Network':network,'Station':station,'Location':location,'Channel':channel,'Start':tr.stats.starttime,'End':tr.stats.endtime}
    avail_rows.append(row)

avail = pd.DataFrame(data=avail_rows)
avail.to_csv('stream_availability.csv')