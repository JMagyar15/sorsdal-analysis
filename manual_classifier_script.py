"""
Script to open plots of events and give input prompts to command line to be answered by viewer

This should take an existing catalogue and write the input information into a new modified catalogue

Flags are given to events that need modifications due to window position (shift of start/end), false detection (delete event), or multiple events (splitting of event into multiple catalogue entries)
"""

from obspy.core import UTCDateTime
from iqvis import data_objects as do
import matplotlib.pyplot as plt
from obspy.core.inventory import inventory
import os
import pandas as pd


c_path = '/Users/jaredmagyar/Documents/SorsdalData/catalogues/bbs'
class_path = '/Users/jaredmagyar/Documents/SorsdalData/classification'
w_path = '/Users/jaredmagyar/Documents/SorsdalData/unprocessed'
stat_path = '/Users/jaredmagyar/Documents/SorsdalData/stations/sorsdal_stations.xml'

section = 3

if section == 1:
    starttime = UTCDateTime(2018,1,8)
    endtime = UTCDateTime(2018,1,8,8)

if section == 2:
    starttime = UTCDateTime(2018,1,20,8)
    endtime = UTCDateTime(2018,1,20,16)

if section == 3:
    starttime = UTCDateTime(2018,1,14,16)
    endtime = UTCDateTime(2018,1,15)


event_cat = do.EventCatalogue(starttime,endtime,c_path)

inv = inventory.read_inventory(stat_path,level='response')
inv = inv.select(station='BBS??')


if not os.path.exists(class_path):
    os.mkdir(class_path)

try:
    class_cat = pd.read_csv(os.path.join(class_path,'classified_catalogue.csv'),index_col=0) 
except:
    class_cat = pd.DataFrame(columns=['event_id','arrival','group'])
    class_cat.to_csv(os.path.join(class_path,'classified_catalogue.csv'))


for event in event_cat:

    #load in the classifications made so far
    class_cat = pd.read_csv(os.path.join(class_path,'classified_catalogue.csv'),index_col=0) 
    #make a new row for the current event
    if event.event_id in class_cat['event_id'].to_numpy():
        print('Event already analysed...moving to next event.')
    
    else:
        event.attach_waveforms(inv,w_path,buffer=2,length=8) #10 second waveforms - can adjust this based on how easy this makes viewing (5 sec may be enough)
        event.remove_response(pre_filt=[1,2,45,50])

        event.attach_waveforms(inv,w_path,buffer=2,length=4)
        event.filter('highpass',freq=1)
        event.context('spectral')
        event.get_spectrograms(32,24)
        event.get_power_spectrum()
        event.context('plot')
        fig = event.new_plotting(plt.figure(figsize=[15,8],layout='compressed'),components=['Z'],spectrogram='right')

        fig.show()
        
        cat_row = event.event_row.drop(['stations','network_time','ref_time','ref_duration'])

        arrival = input('Sector [3NUM/Z/U]')
        group = input('Assigned Group: [NUMBER]') #removed false detection question, just assign as group 11

        cat_row['arrival'] = arrival
        cat_row['group'] = group
        cat_row['event_id'] = cat_row.name

        new_row = cat_row.to_frame().T

        #attach this new row to the previous
        class_cat = pd.concat([class_cat,new_row],ignore_index=True)
        
        #save classifications back to file
        class_cat.to_csv(os.path.join(class_path,'classified_catalogue.csv'))
        plt.close(fig)