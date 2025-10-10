"""
"""

from obspy.core import UTCDateTime
from iqvis import data_objects as do
from iqvis import dayplot_backend as db
from obspy.core.inventory import inventory
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tqdm
import pandas as pd


"""
STA/LTA PARAMETERS
"""
sta = 0.2
lta = 2.0
delta_sta = 50
delta_lta = 50
epsilon = 2
thr_on = 4
thr_off = 3
bbs_sum = 2
ref_sum = 3
avg_wave_speed = 1.5
thr_event_join = 5.0

"""
SWITCHES
"""

detect = True
plotting = False
availability = True

"""
CHUNK AND PATH SET-UP
"""

t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,15)
time_offset = 7
chunk = do.SeismicChunk(t1,t2,time_offset=time_offset)

c_path = '/Users/jaredmagyar/Documents/SorsdalData/catalogues'
stat_path = '/Users/jaredmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'
w_path = '/Users/jaredmagyar/Documents/SorsdalData/unprocessed' #! note that this has been changed to unprocessed waveforms. Filtering with dentrending should hopefully fix this.


if not os.path.exists(c_path):
    os.mkdir(c_path)

bbs_path = os.path.join(c_path,'bbs')
ref_path = os.path.join(c_path,'ref')
ref_un_path = os.path.join(c_path,'ref_un')

if not os.path.exists(bbs_path):
    os.mkdir(bbs_path)

if not os.path.exists(ref_path):
    os.mkdir(ref_path)

if not os.path.exists(ref_un_path):
    os.mkdir(ref_un_path)

"""
STATION SELECTION
"""

total_inv = inventory.read_inventory(stat_path,level='response')
drop = ['REF01','REF03','REF08']

inv = total_inv.copy()
for station in drop:
    inv = inv.remove(station=station)

inv_bbs = inv.select(station='BBS??')
inv_ref = inv.select(station='REF??')

avail_rows = []

"""
DAYPLOT PARAMETERS
"""

dpi = 100
width = 8
height = 10
mins_per_row = 10
page_length = 6*60*60

fig = plt.figure(figsize=(width,height))
bbs_dayplot = db.StackPlot(fig,mins_per_row=mins_per_row,time_offset=time_offset)
ref_dayplot = db.StackPlot(fig,mins_per_row=mins_per_row,time_offset=time_offset)

"""
EVENT DETECTION
"""

if detect:
    for daychunk in chunk(24*60*60):
        
        if availability:
            #firstly get the data availability
            print('Recording data availability for',daychunk.str_name)
            daychunk.attach_waveforms(total_inv,w_path)
            split_stream = daychunk.stream.split()

            for tr in split_stream:
                network, station, location, channel = tr.id.split('.')
                row = {'Network':network,'Station':station,'Location':location,'Channel':channel,'Start':tr.stats.starttime,'End':tr.stats.endtime}
                row = pd.DataFrame(data=row,index=[tr.id])
                avail_rows.append(row)


        #now make the broadband catalogue
        daychunk.attach_waveforms(inv_bbs,w_path,buffer=60*60) #hour long buffer for filtering and STA/LTA
        daychunk.remove_response(pre_filt=[1,2,45,50],taper=False)   

        daychunk.context('detect')
        daychunk.detect_events(bbs_path,trigger_type='multistalta',sta=sta,lta=lta,delta_sta=delta_sta,delta_lta=delta_lta,epsilon=epsilon,thr_on=thr_on,thr_off=thr_off,thr_coincidence_sum=bbs_sum,avg_wave_speed=avg_wave_speed,thr_event_join=thr_event_join) 
        

        #unprocessed short-period catalogue
        daychunk.attach_waveforms(inv_ref,w_path,buffer=60*60) #hour long buffer for filtering and STA/LTA
        daychunk.detect_events(ref_un_path,trigger_type='multistalta',sta=sta,lta=lta,delta_sta=delta_sta,delta_lta=delta_lta,epsilon=epsilon,thr_on=thr_on,thr_off=thr_off,thr_coincidence_sum=ref_sum,avg_wave_speed=avg_wave_speed,thr_event_join=thr_event_join) 

        #processed short-period catalogue
        daychunk.attach_waveforms(inv_ref,w_path,buffer=60*60) #hour long buffer for filtering and STA/LTA
        daychunk.decimate(10)
        daychunk.remove_response(pre_filt=[1,2,45,50],taper=False)

        daychunk.context('detect')
        daychunk.detect_events(ref_path,trigger_type='multistalta',sta=sta,lta=lta,delta_sta=delta_sta,delta_lta=delta_lta,epsilon=epsilon,thr_on=thr_on,thr_off=thr_off,thr_coincidence_sum=ref_sum,avg_wave_speed=avg_wave_speed,thr_event_join=thr_event_join) 
        
    
    if availability:
        avail = pd.concat(avail_rows,ignore_index=True)
        avail.to_csv('stream_availability.csv')
    
    
    # for page_chunk in chunk(6*60*60):
    #     bbs_cat = do.EventCatalogue(page_chunk.starttime,page_chunk.endtime,bbs_path)
    #     bbs_cat.events['group'] = 2
    #     ref_cat = do.EventCatalogue(page_chunk.starttime,page_chunk.endtime,ref_path)
    #     ref_cat.events['group'] = 7
    #     ref_un_cat = do.EventCatalogue(page_chunk.starttime,page_chunk.endtime,ref_un_path)
    #     ref_un_cat.events['group'] = 8

    #     combined_cat = (bbs_cat + ref_cat) + ref_un_cat

    #     page_chunk.attach_waveforms(inv_bbs,w_path)
    #     bbs_dayplot.add_events(page_chunk,combined_cat)

    #     page_chunk.attach_waveforms(inv_ref,w_path)
    #     page_chunk.decimate(10)
    #     ref_dayplot.add_events(page_chunk,combined_cat)

    # bbs_dayplot.normalise_traces()
    # ref_dayplot.normalise_traces()

    # for tr_id in inv_bbs.get_contents()['channels']:
    #     #make a pdf document for this channel
    #     filename = os.path.join(bbs_path,'dayplot__' + tr_id + '__' + chunk.str_name + '.pdf')
    #     p = PdfPages(filename) #set up the object used to save each channel as page in a pdf
    #     for page_chunk in chunk(page_length):
    #         fig, ax = bbs_dayplot.make_page(tr_id,page_chunk)
    #         fig.savefig(p,format='pdf')
    #     p.close()

    # for tr_id in inv_ref.get_contents()['channels']:
    #     #make a pdf document for this channel
    #     filename = os.path.join(ref_path,'dayplot__' + tr_id + '__' + chunk.str_name + '.pdf')
    #     p = PdfPages(filename) #set up the object used to save each channel as page in a pdf
    #     for page_chunk in chunk(page_length):
    #         fig, ax = ref_dayplot.make_page(tr_id,page_chunk)
    #         fig.savefig(p,format='pdf')
    #     p.close()

if plotting:
    small_bbs_cat = do.EventCatalogue(t1,t1+10*60,bbs_path)
    small_ref_cat = do.EventCatalogue(t1,t1+10*60,ref_path)

    #attached all the waveforms this time around to see if broadbands pick up ref, and ref pick up broadbands.

    filename = 'bbs_catalogue.pdf'
    p = PdfPages(os.path.join(bbs_path,filename))
    for event in tqdm.tqdm(small_bbs_cat):
        fig = plt.figure(figsize=(6,12),constrained_layout=True)
        event.attach_waveforms(total_inv,w_path,buffer=1,length=4)
        print(event.stream)
        event.remove_response(pre_filt=[1,2,45,50])
        #event.filter('bandpass',freqmin=1,freqmax=50)
        event.context('plot')
        event.attach_fig(fig,bbs_path)
        fig = event.plot_event(components=['Z'],stack='stations',spectrogram=None)
        fig.savefig(p,format='pdf',dpi=300)
    p.close()

    filename = 'ref_catalogue.pdf'
    p = PdfPages(os.path.join(ref_path,filename))
    for event in tqdm.tqdm(small_ref_cat):
        fig = plt.figure(figsize=(6,12),constrained_layout=True)
        event.attach_waveforms(total_inv,w_path,buffer=1,length=4)
        event.remove_response(pre_filt=[1,2,45,50])
        #event.filter('bandpass',freqmin=1,freqmax=50)
        event.context('plot')
        event.attach_fig(fig,ref_path)
        fig = event.plot_event(components=['Z'],stack='stations',spectrogram=None)
        fig.savefig(p,format='pdf',dpi=300)
    p.close()