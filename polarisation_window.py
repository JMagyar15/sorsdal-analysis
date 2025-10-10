import numpy as np
from obspy.core.inventory import inventory
from iqvis import data_objects as do
from obspy.core import UTCDateTime
import tqdm
import os
import glob
import pandas as pd
from iqvis import spatial_analysis as sa
from iqvis import event_calculation as ec

all_events = False
group_events = False

localisation = True

local_path = '/Users/jmagyar/Documents/SorsdalData'
cloud_path = '/Users/jmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes'

c_path = os.path.join(cloud_path,'sorsdal_catalogues')
stat_path = os.path.join(local_path,'stations/sorsdal_stations.xml')
w_path = os.path.join(local_path,'waveforms')
p_path = cloud_path
class_path = c_path


inv = inventory.read_inventory(stat_path,level='response')
ref_inv = inv.select(station='REF??')
bbs_inv = inv.select(station='BBS??')


t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,1)

chunk = do.SeismicChunk(t1,t2)

#get the backazimuth and radial distance estimates from the mfp results.
mfp = chunk.load_csv(c_path,'mfp')
mfp['baz'] *= (np.pi/180) #convert to radians.


#grid here doesn't matter, just need the centre to get the backazimuth from the stations.
backazimuth_grid = np.linspace(0,2*np.pi,180)
radial_grid = np.logspace(-2,1.3,100,base=10)
slowness_grid = np.linspace(0.2,1.0,30)

grid, centre = sa.radial_grid(backazimuth_grid,radial_grid,slowness_grid,ref_inv)

chunk = do.SeismicChunk(t1,t2)

if all_events:

    for daychunk in chunk:
        event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)
        pol_cat = event_cat.attributes.drop(labels='group',axis=1)

        filename = os.path.join(c_path,'polarisation_attributes__' + daychunk.str_name + '.csv')
        
        for event in tqdm.tqdm(event_cat,total=event_cat.N):

            event.attach_waveforms(bbs_inv,w_path,buffer=2,length=4)
            event.filter('highpass',freq=1)
            event.remove_sensitivity()
            event.context('spectral')
            event.get_power_spectrum()
            window = event.get_data_window()

            amps = []
            codes = []
            for net in event.inv:
                for sta in net:
                    sta_stream = window.select(network=net.code,station=sta.code)
                    flattened = np.stack([tr.data.astype(np.float64) for tr in sta_stream],axis=0)
                    amp = np.linalg.norm(flattened,axis=1) #otherwise compute energy here as squared norm along axis 0.
                    max_amp = np.max(amp)
                    amps.append(max_amp)
                    codes.append((net.code,sta.code))
            
            
            ind = np.nanargmax(amps)

            rad = mfp['rad'][event.event_id]
            baz = mfp['baz'][event.event_id]

            
            station_baz = sa.centre_baz_to_station_baz(baz,rad,bbs_inv,centre)
            max_sta_baz = station_baz[ind] * (180/np.pi)

            if max_sta_baz < 0:
                max_sta_baz += 360

            #now that we have the backazimiuth of the source relative to the station with the greatest amplitude, calculate polarisation attributes
            event.stream = event.stream.select(network=codes[ind][0],station=codes[ind][1])

            rayleigh_corr, p_corr, s_corr = ec.event_polarisation(event,max_sta_baz)
        
            pol_cat.at[event.event_id,'R_corr'] = rayleigh_corr[0]
            pol_cat.at[event.event_id,'P_corr'] = p_corr[0]
            pol_cat.at[event.event_id,'S_corr'] = s_corr[0]
        
        pol_cat.to_csv(filename)


if group_events:
    
    event_cat = do.EventCatalogue(t1,t2,c_path)

    xcorr_files = glob.glob(os.path.join(c_path,'xcorr*'))
    xcorr_files.sort()
    xcorr = pd.concat([pd.read_csv(file,index_col=0) for file in xcorr_files],ignore_index=False)

    threshold_dict = {'1':0.8,
            '2':0.75,
            '3':0.65,
            '4':0.75,
            '5':0.75,
            '6':0.75,
            '7':0.65,
            '8':0.65}


    for i, row in xcorr.iterrows():
        for temp_name, xc in row.items():
            group = temp_name[:-1]
            thresh = threshold_dict[group]
            row.at[temp_name] = xc / thresh
        max_col = row.argmax()
        val = row.iloc[max_col]
        temp = xcorr.columns[max_col]
        xcorr.at[i,'max_xcorr'] = val
        if val > 1.0:
            xcorr.at[i,'template'] = temp
            xcorr.at[i,'group'] = temp[:-1]
        else:
            xcorr.at[i,'template'] = '0' #index for unmatched signals.
            xcorr.at[i,'group'] = '0'

    event_cat.add_classification(xcorr.reset_index())

    temp_groups = event_cat.group_split()

    for group, group_cat in temp_groups.items():
        if int(group) in [1,2,3,4,5,6,7,8]:

            print('Polarisation for group',group)
            filename = os.path.join(c_path,'polarisation_attributes__group__' + group + '.csv')
            pol_cat = group_cat.attributes.drop(labels='group',axis=1)

            for event in tqdm.tqdm(group_cat,total=group_cat.N):
                event.attach_waveforms(bbs_inv,w_path,buffer=2,length=4)
                event.filter('highpass',freq=1)
                event.remove_sensitivity()
                event.context('spectral')
                event.get_power_spectrum()
                window = event.get_data_window()

                amps = []
                codes = []
                for net in event.inv:
                    for sta in net:
                        sta_stream = window.select(network=net.code,station=sta.code)
                        flattened = np.stack([tr.data.astype(np.float64) for tr in sta_stream],axis=0)
                        amp = np.linalg.norm(flattened,axis=1) #otherwise compute energy here as squared norm along axis 0.
                        max_amp = np.max(amp)
                        amps.append(max_amp)
                        codes.append((net.code,sta.code))
                
                
                ind = np.nanargmax(amps)

                rad = mfp['rad'][event.event_id]
                baz = mfp['baz'][event.event_id]

                
                station_baz = sa.centre_baz_to_station_baz(baz,rad,bbs_inv,centre)
                max_sta_baz = station_baz[ind] * (180/np.pi)

                if max_sta_baz < 0:
                    max_sta_baz += 360

                #now that we have the backazimiuth of the source relative to the station with the greatest amplitude, calculate polarisation attributes
                event.stream = event.stream.select(network=codes[ind][0],station=codes[ind][1])

                rayleigh_corr, p_corr, s_corr = ec.event_polarisation(event,max_sta_baz)
            
                pol_cat.at[event.event_id,'R_corr'] = rayleigh_corr[0]
                pol_cat.at[event.event_id,'P_corr'] = p_corr[0]
                pol_cat.at[event.event_id,'S_corr'] = s_corr[0]
            
            pol_cat.to_csv(filename)


if localisation:
    backazimuth_grid = np.linspace(0,2*np.pi,101)[:-1]
    radial_grid = np.logspace(-5,4,50,base=2)

    baz, centre = sa.polarisation_grid(backazimuth_grid,radial_grid,bbs_inv) #later still want to use REF for consistancy
    
    for daychunk in chunk:
        event_cat = do.EventCatalogue(daychunk.starttime,daychunk.endtime,c_path)
        pol_cat = event_cat.attributes.drop(labels='group',axis=1)

        filename = os.path.join(c_path,'polarisation_localisation_duration__' + daychunk.str_name + '.csv')
        
        for event in tqdm.tqdm(event_cat,total=event_cat.N):

            if event.window_length <= 30:
                event.attach_waveforms(bbs_inv,w_path,buffer=2)
            else:
                event.attach_waveforms(bbs_inv,w_path,buffer=2,length=30) #cap length at 30s
            event.remove_response(pre_filt=[1,2,45,50])
            event.context('polarisation')

            R_corr, total_R_corr, P_corr, total_P_corr = event.correlation_power(baz)

            ind = np.argmax(total_R_corr)
            ind = np.unravel_index(ind,total_R_corr.shape)

            baz_loc = backazimuth_grid[ind[0]]
            rad_loc = radial_grid[ind[1]]

            pol_cat.at[event.event_id,'R_baz'] = np.rad2deg(baz_loc)
            pol_cat.at[event.event_id,'R_rad'] = rad_loc
            pol_cat.at[event.event_id,'R_corr'] = total_R_corr[ind[0],ind[1]]

            ind = np.argmax(total_P_corr)
            ind = np.unravel_index(ind,total_P_corr.shape)

            baz_loc = backazimuth_grid[ind[0]]
            rad_loc = radial_grid[ind[1]]

            pol_cat.at[event.event_id,'P_baz'] = np.rad2deg(baz_loc)
            pol_cat.at[event.event_id,'P_rad'] = rad_loc
            pol_cat.at[event.event_id,'P_corr'] = total_P_corr[ind[0],ind[1]]

            pol_cat.at[event.event_id,'N'] = R_corr.shape[-1]

        
        pol_cat.to_csv(filename)
