
from obspy.core import UTCDateTime
from cryoquake import data_objects as do
from obspy.core.inventory import inventory
import os
from pathlib import Path


psd_window = 5
psd_overlap = 0.5

t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,15)

root = Path(__file__).parent.parent
w_path = root / "waveforms"
spec_path = root / "spectrograms"
s_file = root / "stations" / "sorsdal_stations.xml"


for path in [spec_path]:
    if not os.path.exists(path):
        os.mkdir(path)

inv_bbs = inventory.read_inventory(s_file,level='response').select(station='BBS??')
inv_ref = inventory.read_inventory(s_file,level='response').select(station='REF??')
inv = inv_bbs + inv_ref
chunk = do.SeismicChunk(t1,t2,time_offset=7)

for daychunk in chunk:
    print('Computing PSDs for',daychunk.str_name)

    daychunk.context('spectral')
    daychunk.attach_waveforms(inv_bbs,w_path)#,buffer=60*60)
    daychunk.make_periodograms(psd_window,psd_overlap,spec_path)

    daychunk.context('spectral')
    daychunk.attach_waveforms(inv_ref,w_path)#,buffer=60*60)
    daychunk.decimate(10)
    daychunk.make_periodograms(psd_window,psd_overlap,spec_path)
