"""
MOVE PLOTTING OF CATALOGUE HERE - MORE EASY TO MODIFY AND BRING OTHER RESULTS IN (E.G. AUTOENCODER CLUSTERING)
"""

from obspy.core import UTCDateTime
from iqvis import data_objects as do
from obspy.core.inventory import inventory
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import pandas as pd
from obspy.imaging.util import _set_xaxis_obspy_dates

from matplotlib import rc
import matplotlib.font_manager as fm

"""
SWITCHES
"""
plot_spec = True
plot_att = False

rc('text', usetex=True)
rc('font', size=8)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Optima']})

one_col = 3.35 #inches
two_col = 7.0 #inches


t1 = UTCDateTime(2018,1,1)
t2 = UTCDateTime(2018,2,15) #for now, just test with a couple of days to check all is working
ref_time = UTCDateTime(2018,1,10) #for finding the response information


local_path = '/Users/jmagyar/Documents/SorsdalData'
cloud_path = '/Users/jmagyar/Library/Mobile Documents/com~apple~CloudDocs/Outputs/Icequakes'

c_path = os.path.join(cloud_path,'sorsdal_catalogues')
stat_path = os.path.join(local_path,'stations/sorsdal_stations.xml')
env_path = os.path.join(local_path,'environmental')
outplots = os.path.join(cloud_path,'catalogue_plots')
spec_path = os.path.join(local_path,'spectrograms')


for path in [c_path,spec_path,outplots]:
    if not os.path.exists(path):
        os.mkdir(path)


chunk = do.SeismicChunk(t1,t2,time_offset=7) #local time is UTC + 07:00

avail = pd.read_csv('stream_availability.csv')
event_cat = do.EventCatalogue(t1,t2,c_path) #get the event catalogue

inv = inventory.read_inventory(stat_path,level='response')

if plot_spec:

    for network in inv.select(station='BBS??',channel='??Z'):#.select(station='BBS??'):
        for station in network:
            for channel in station:
                code = network.code + '.' + station.code + '.' + channel.location_code + '.' + channel.code
                print('Plotting',code)
                """
                PLOTTING OF THE BINNED SPECTROGRAMS
                """
                full = np.load(os.path.join(spec_path,'full_spectrogram__' + code + '__' + chunk.str_name + '.npz'),allow_pickle=True)
                daily = np.load(os.path.join(spec_path,'diurnal_spectrogram__' + code + '__' + chunk.str_name + '.npz'),allow_pickle=True)
                tidal = np.load(os.path.join(spec_path,'tide_phase_spectrogram__' + code + '__' + chunk.str_name + '.npz'),allow_pickle=True)
                
                resp = inv.get_response(code,ref_time)
                sens = abs(resp.get_evalresp_response_for_frequencies(full['f']))

                """
                Full chunk plotting
                """

                f_band = full['f'][(full['f']>3)&(full['f']<15)]
                spec_band = full['spec'][(full['f']>3)&(full['f']<15),:]
                spec_band /= sens[(full['f']>3)&(full['f']<15),None] **2
                med_amp = np.sqrt(trapezoid(spec_band,f_band,axis=0))

                fig = plt.figure(figsize=(two_col,two_col/1.4))
                grid_spec = fig.add_gridspec(nrows=4,ncols=2,width_ratios=[1,0.02],height_ratios=[1,0.6,0.4,0.2],wspace=0.04,hspace=0.35)

                start_mt = chunk.starttime.matplotlib_date
                end_mt = chunk.endtime.matplotlib_date

                #firstly deal with the spectrogram + colourbar in the top row.
                spec_ax = fig.add_subplot(grid_spec[0,0])
                spec_cb_ax = fig.add_subplot(grid_spec[0,1])

                t_mt = [time.matplotlib_date for time in full['t']]

                db_spec = 10 * np.log10(full['spec'][1:,:]) - 20 * np.log10(sens)[1:,None]

                spec_plot = spec_ax.pcolormesh(t_mt,full['f'][1:],db_spec,cmap='PuBu',vmax=-120,vmin=-200)
                spec_ax.hlines([3,15],xmin=t_mt[0],xmax=t_mt[-1],color='black',ls='--',lw=0.8)
                cb = plt.colorbar(spec_plot,cax=spec_cb_ax,label='Amplitude\n' +  r'(m$^2$s$^{-2}$Hz$^{-1}$) (dB)')

                spec_ax.set_yscale('log')
                spec_ax.set_ylabel('Frequency\n(Hz)')

                #environmental data and band-integrated median seismic amplitude
                time_series_ax = fig.add_subplot(grid_spec[1,0],sharex=spec_ax)

                tide_ax = plt.twinx(time_series_ax)
                time_series_ax.hlines(0,xmin=start_mt,xmax=end_mt,color='grey',ls='--',lw=0.5)

                tide_ax.plot(t_mt,full['tide'],c='cornflowerblue',lw=0.8)
                tide_ax.set_ylabel('Tide Height\n(m)')

                time_series_ax.plot(t_mt,full['temp'],c='indianred',lw=0.8)
                time_series_ax.set_ylabel('Temperature\n' + r'($^\circ$C)')

                #histogram of event detections
                hist_ax = fig.add_subplot(grid_spec[2,0],sharex=spec_ax)


                hist_ax.bar(t_mt,height=full['events'],width=t_mt[1]-t_mt[0],bottom=0,align='center',color='grey',alpha=0.5)
                hist_ax.set_ylabel('Event\nCount')

                amp_ax = plt.twinx(hist_ax)
                amp_ax.plot(t_mt,med_amp,color='black',lw=0.8)
                amp_ax.set_ylim(bottom=0)


                amp_ax.set_ylabel('Amplitude\n' + r'(m s$^{-1}$)')
                amp_ax.yaxis.get_offset_text().set_position((1.07,1))

                #data availability for the broadband stations
                avail_ax = fig.add_subplot(grid_spec[3,0],sharex=spec_ax)

                stations = ['BBS05','BBS06','BBS09']

                rows = []

                for sta_code in stations:
                    channel = 'HHZ'
                    subset = avail[(avail['Station']==sta_code) & (avail['Channel']==channel)]
                    
                    starts = subset['Start'].to_numpy()
                    ends = subset['End'].to_numpy()

                    short_start = []
                    short_end = []
                    short_start.append(starts[0])

                    for i, start in enumerate(starts[1:]):
                        if start != ends[i]:
                            short_start.append(start)
                            short_end.append(ends[i])

                    short_end.append(ends[-1])

                    for i in range(len(short_start)):
                        rows.append({'Station':sta_code,'Start':short_start[i],'End':short_end[i]})

                df = pd.DataFrame(data=rows)

                i = 0
                labels = []
                coords = []
                for net in inv.select(station='BBS??',channel='??Z'):
                    for sta in net:
                        for ch in sta:
                            i -= 1
                            subset = df[(df['Station']==sta.code)]
                            start = np.array([UTCDateTime(time).matplotlib_date for time in subset['Start']])#pd.to_datetime(subset['Start'])# + pd.Timedelta(hours=chunk.time_offset)
                            end = np.array([UTCDateTime(time).matplotlib_date for time in subset['End']])#pd.to_datetime(subset['End']) #+ pd.Timedelta(hours=chunk.time_offset)
                            duration = end - start

                            
                            avail_ax.barh(i,duration,height=0.7,left=start,facecolor='rosybrown',edgecolor='black',lw=0.8)
                            coords.append(i)
                            labels.append(sta.code)

                avail_ax.set_yticks(coords)
                avail_ax.set_yticklabels(labels)
                avail_ax.spines['right'].set_visible(False)
                avail_ax.spines['left'].set_visible(False)
                avail_ax.spines['top'].set_visible(False)
                avail_ax.spines['bottom'].set_visible(False)

                avail_ax.set_xlim((start_mt,end_mt))
                _set_xaxis_obspy_dates(avail_ax)
                plt.setp(spec_ax.get_xticklabels(), visible=False)
                plt.setp(time_series_ax.get_xticklabels(), visible=False)
                plt.setp(hist_ax.get_xticklabels(), visible=False)

                sign = '%+i' % chunk.time_offset
                sign = sign[0]
                #label = "UTC (Local Time = UTC %s %02i:%02i)" % (sign, abs(chunk.time_offset),(chunk.time_offset % 1 * 60))
                label = "UTC (Local Solar Time = UTC + 05:12)" #% (sign, abs(chunk.time_offset),(chunk.time_offset % 1 * 60))

                avail_ax.set_xlabel(label)

                spec_ax.annotate('(a)',(0,1.05),xycoords='axes fraction')
                time_series_ax.annotate('(b)',(0,1.07),xycoords='axes fraction')
                hist_ax.annotate('(c)',(0,1.1),xycoords='axes fraction')
                avail_ax.annotate('(d)',(0,1.14),xycoords='axes fraction')

                plot_name = os.path.join(outplots,'full_spectrogram__' + code + '__' + chunk.str_name + '.png')
                plot_name_eps = os.path.join(outplots,'full_spectrogram__' + code + '__' + chunk.str_name + '.eps')
                plot_name_pdf = os.path.join(outplots,'full_spectrogram__' + code + '__' + chunk.str_name + '.pdf')

                fig.savefig(plot_name,bbox_inches='tight')
                fig.savefig(plot_name_eps,bbox_inches='tight')
                fig.savefig(plot_name_pdf,dpi=400,bbox_inches='tight')

                """
                WRAPPED PLOTS
                """
                
                f_band = daily['f'][(daily['f']>3)&(daily['f']<15)]
                spec_band = daily['spec'][(daily['f']>3)&(daily['f']<15),:]
                spec_band /= sens[(daily['f']>3)&(daily['f']<15),None]**2
                med_amp = np.sqrt(trapezoid(spec_band,f_band,axis=0))

                fig = plt.figure(figsize=(two_col,two_col/2.5))
                grid_spec = fig.add_gridspec(nrows=3,ncols=3,width_ratios=[0.5,0.5,0.02],height_ratios=[1,0.3,0.7],wspace=0.1,hspace=0.3)

                #firstly deal with the spectrogram + colourbar in the top row.
                day_spec_ax = fig.add_subplot(grid_spec[0,0])
                #day_spec_ax.set_title('Diurnal Wrapping')
                tide_spec_ax = fig.add_subplot(grid_spec[0,1],sharey=day_spec_ax)
                #tide_spec_ax.set_title('Tidal Wrapping')
                spec_cb_ax = fig.add_subplot(grid_spec[0,2])

                t_mt = [time.matplotlib_date for time in daily['t']]

                db_spec = 10 * np.log10(daily['spec'][1:,:]) - 20 * np.log10(sens)[1:,None]

                spec_plot = day_spec_ax.pcolormesh(t_mt,daily['f'][1:],db_spec,cmap='PuBu',vmin=-200,vmax=-120)
                day_spec_ax.hlines([3,15],xmin=t_mt[0],xmax=t_mt[-1],color='black',ls='--',lw=0.8)
                cb = plt.colorbar(spec_plot,cax=spec_cb_ax,label='Amplitude\n' +  r'(m$^2$s$^{-2}$Hz$^{-1}$) (dB)')

                day_spec_ax.set_yscale('log')
                day_spec_ax.set_ylabel('Frequency\n(Hz)')

                #environmental data and band-integrated median seismic amplitude
                temp_ax = fig.add_subplot(grid_spec[1,0],sharex=day_spec_ax)


                temp_ax.hlines(0,xmin=t_mt[0],xmax=t_mt[-1],color='grey',ls='--',lw=0.5)

                temp_ax.plot(t_mt,daily['temp'],c='indianred',lw=0.8)
                temp_ax.set_ylabel('Temperature \n' +  r'($^\circ$C)')


                #histogram of event detections
                day_hist_ax = fig.add_subplot(grid_spec[2,0],sharex=day_spec_ax)

                day_hist_ax.bar(t_mt,height=daily['events'],width=t_mt[1]-t_mt[0],bottom=0,align='center',color='grey',alpha=0.5)
                day_hist_ax.set_ylabel('Event\nCount')

                day_amp_ax = plt.twinx(day_hist_ax)
                day_amp_ax.plot(t_mt,med_amp,color='black',lw=0.8)
                day_amp_ax.set_ylim(bottom=0)


                plt.setp(day_spec_ax.get_xticklabels(), visible=False)
                plt.setp(temp_ax.get_xticklabels(), visible=False)

                from matplotlib.dates import DateFormatter, HourLocator

                myFmt = DateFormatter("%H:%M")
                day_hist_ax.xaxis.set_major_formatter(myFmt)
                day_hist_ax.xaxis.set_major_locator(HourLocator(interval=8))


                sign = '%+i' % chunk.time_offset
                sign = sign[0]
                #label = "UTC (Local Time = UTC %s %02i:%02i)" % (sign, abs(chunk.time_offset),(chunk.time_offset % 1 * 60))
                label = "UTC (Local Solar Time = UTC + 05:12)"
                day_hist_ax.set_xlabel(label)


                f_band = tidal['f'][(tidal['f']>3)&(tidal['f']<15)]
                spec_band = tidal['spec'][(tidal['f']>3)&(tidal['f']<15),:]
                spec_band /= sens[(tidal['f']>3)&(tidal['f']<15),None]**2
                med_amp = np.sqrt(trapezoid(spec_band,f_band,axis=0))

    
                db_spec = 10 * np.log10(tidal['spec'][1:,:]) - 20 * np.log10(sens)[1:,None]

                spec_plot = tide_spec_ax.pcolormesh(tidal['phase'],tidal['f'][1:],db_spec,cmap='PuBu',vmin=-200,vmax=-120)
                tide_spec_ax.hlines([3,15],xmin=tidal['phase'][0],xmax=tidal['phase'][-1],color='black',ls='--',lw=0.8)

                spec_ax.set_yscale('log')

                #environmental data and band-integrated median seismic amplitude
                tide_ax = fig.add_subplot(grid_spec[1,1],sharex=tide_spec_ax)

                tide_ax.plot(tidal['phase'],tidal['tide'],c='cornflowerblue',lw=0.8)
                tide_ax.set_ylabel('Tide\n(m)')
                tide_ax.yaxis.tick_right()
                tide_ax.yaxis.set_label_position('right')


                #histogram of event detections
                tide_hist_ax = fig.add_subplot(grid_spec[2,1],sharex=tide_spec_ax,sharey=day_hist_ax)
                tide_hist_ax.bar(tidal['phase'],height=tidal['events'],width=tidal['phase'][1]-tidal['phase'][0],bottom=0,align='center',color='grey',alpha=0.5)
                tide_hist_ax.set_xlabel('Tidal Phase (degrees)')

                tide_amp_ax = plt.twinx(tide_hist_ax)
                tide_amp_ax.plot(tidal['phase'],med_amp,color='black',lw=0.8)
                tide_amp_ax.set_ylabel('Amplitude\n' + r'(m s$^{-1}$)')
                tide_amp_ax.set_ylim(bottom=0)
                tide_amp_ax.yaxis.get_offset_text().set_position((1.15,1))
                plt.setp(day_amp_ax.get_yaxis().get_offset_text(), visible=False)



                # day_max = day_amp_ax.get_ylim()[1]
                # tide_max = tide_amp_ax.get_ylim()[1]
                # ymax = max(day_max,tide_max)

                # #cannot seem to sharey for twinaxes, so easiest just to manually set them with the same yaxis limits.
                # tide_amp_ax.set_ylim((0,ymax))
                # day_amp_ax.set_ylim((0,ymax))

                tide_amp_ax.sharey(day_amp_ax)


                plt.setp(tide_spec_ax.get_xticklabels(), visible=False)
                plt.setp(tide_ax.get_xticklabels(), visible=False)
                plt.setp(tide_hist_ax.get_yticklabels(), visible=False)
                plt.setp(day_amp_ax.get_yticklabels(),visible=False)
                plt.setp(tide_spec_ax.get_yticklabels(),visible=False)


                plot_name = os.path.join(outplots,'wrapped_spectra__' + code + '__' + chunk.str_name + '.png')
                plot_name_eps = os.path.join(outplots,'wrapped_spectra__' + code + '__' + chunk.str_name + '.eps')
                plot_name_pdf = os.path.join(outplots,'wrapped_spectra__' + code + '__' + chunk.str_name + '.pdf')

                fig.savefig(plot_name,dpi=400,bbox_inches='tight')
                fig.savefig(plot_name_eps,bbox_inches='tight')
                fig.savefig(plot_name_pdf,dpi=400,bbox_inches='tight')

