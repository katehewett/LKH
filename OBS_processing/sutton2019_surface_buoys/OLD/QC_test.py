'''
This code is to run basic QC on NOAA data products for 3 moornings: 
    Chaba 
    Cape Elizabeth 
    Cape Arago 

(see Sutton et al 2014 + 2019) 

We first ran process_webdata.py which saves iput .nc files here: 
/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/py_files/

Notes after plotting processed data:
(1) Data provided is not 3 hourly (it's different dt's and gappy)
(2) Spikes are also present in the data - 
    and sometimes it looks like as senor may have fouled (?) 
(3) It looks like, but need to verify, that the data presented passed a gross range test 
or equiv QC, but not others? Need to verify?
Going to run some simple QC to flag suspect data

We ran thru this exercise so we could compare with lowpass mooring extractions.

TODO: 1) add stn and DTmax as line arguments 


'''

testing = True 

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
import datetime 
from datetime import timedelta
import statistics
from ioos_qc import qartod

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

Ldir = Lfun.Lstart()
    
# processed data location
source = 'ocnms'
otype = 'moor' 
in_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'py_files'
# out_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'daily'
# fig_out_dir = Ldir['parent'] / 'LKH_output' / 'sutton2019_surface_buoys' / 'plots' / 'daily_avg_processing'

#if os.path.exists(out_dir)==False:
#    Lfun.make_dir(out_dir, clean = False)
#if os.path.exists(fig_out_dir)==False:
#    Lfun.make_dir(fig_out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

# This is a little lazy, but you need to enter the station name below with the list here:
# update to keyboard enter sn_name_dict and DTmax
#sn_name_dict = {
#    'CHABA':'Chaba',
#    'CAPEELIZABETH':'Cape Elizabeth', # less varialbes
#    'CAPEARAGO':'Cape Arago',
#}

sn_name_dict = {
    'CHABA':'Chaba'
} 
sn_list = list(sn_name_dict.keys())

#cdx = 1 

sn = sn_list[0]
print(sn)
    
fn_in = posixpath.join(in_dir, (sn +'.nc'))
#fn_out = posixpath.join(out_dir, (sn + '_daily.nc'))

ds = xr.open_dataset(fn_in, decode_times=True)  

# Step 1: id deployment chunks using DTmax = 12 hours
# DTmax = 15 enter as line argument later
DTmax = 15 

dt = pd.to_datetime(ds.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df = pd.DataFrame({'date':dt.normalize(),'time_utc':dt}) 
df['time_diff'] = df['time_utc'].diff()
df['LT_DTmax'] = df['time_utc'].diff()< pd.Timedelta(hours=DTmax)

# find start & stop indicies of all continious data        
s = pd.Series(df['LT_DTmax'])
grp = s.eq(False).cumsum()
arr = grp.loc[s.eq(True)].groupby(grp).apply(lambda x: [x.index.min()-1,x.index.max()])
a = np.array(arr) # its an array of start and stop indicies
                  # a[0] = index of first chunk of data: (start,end) aka (a[0][0],a[0][1]) 
                                       
var_list = list(ds.keys())

window_size = 30 # for rolling stats 
#min_samples = 20 
            
for vn in var_list: 
    # Cape Elizabeth has less variables than the other 2. E.g no DOXY; DO 
    # Go thru exercise if there are values (note: if all = True; vn.values = all nans) 
    if np.all(np.isnan(ds[str(vn)].values))==False:
        
        #fig_out = posixpath.join(fig_out_dir,(sn+'_'+vn+'_DailyAvg.png'))
        df2 = df.copy() #initialize new dataframe for each var
        df2['time_f']=pd.NaT
        df2[str(vn)]=np.nan
        df2[str(vn+'_outliers')]=np.nan
        
        for idx in range(0,np.shape(a)[0]): # step thru each chunks of continious data from the web
            itime = dt[a[idx][0]:a[idx][-1]]
            ivar = pd.Series(ds[vn].values[a[idx][0]:a[idx][-1]])
            
            qc_results = qartod.spike_test(inp=ivar,suspect_threshold=0.9,fail_threshold=2,)
            '''  GOOD = 1
                 UNKNOWN = 2
                 SUSPECT = 3
                 FAIL = 4
                 MISSING = 9 '''
            
            df3 = pd.DataFrame({'time_utc':itime,'var':ivar,'QC':qc_results}) 
            df3[str(vn + '_spike1')] = np.nan
            df3['GOOD'] = qc_results==1
            df3['GOOD'].astype(int)
            df3.loc[qc_results==1, 'SA_spike1'] = ivar[qc_results==1]
            
            # plot checks
            fs=12
            plt.rc('font', size=fs)
            fig = plt.figure(figsize=(16,8))
            ax0 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
            #plt.plot(dt,ds[vn].values,color = 'black',marker='.',linestyle='-',linewidth=1)
            ax0.set_ylabel(str(vn)+' '+str(ds[str(vn)].attrs['units']))
            ax0.set_title(str(sn)+': sutton et al. 2019 data (imported) chunk: ' + str(idx))
            plt.grid(True)
            
            plt.plot(itime,ivar,color = 'black',marker='.',linestyle='-',linewidth=1)
            
            fstdev = np.floor(np.nanstd(ivar))
            ymin = np.floor(np.nanmin(ivar))-fstdev
            ymax = np.ceil(np.nanmax(ivar))+fstdev
            yticks = np.linspace(ymin,ymax,5)
            ax0.set_ylim([ymin,ymax])
            ax0.set_yticks(yticks)
            ax0.set_xlim([itime[0],itime[-1]])
            
            ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
            plt.plot(itime,qc_results,color = 'black',marker='.',linestyle='none',linewidth=1)
            ax1.set_xlim([itime[0],itime[-1]])
            ax1.set_ylabel('QC vals Spike Test')
            
            ax2 = plt.subplot2grid((3,1),(2,0),colspan=1,rowspan=1)
            plt.plot(itime,df3['SA_spike1'],color = 'blue',marker='.',linestyle='none',linewidth=1)
            ax2.set_ylim([ymin,ymax])
            ax2.set_yticks(yticks)
            ax2.set_xlim([itime[0],itime[-1]])
            
            
            sys.exit()
            
            
            
    daily_stats = df.groupby('date').agg(daily_avg=(str(vn), 'mean'), count=(str(vn),'count'))
    time_helper = df.groupby('date').agg(min_time = ('time_utc','min'), max_time = ('time_utc','max'))
    time_helper['DT'] = time_helper['max_time']-time_helper['min_time']
    
    if np.all(np.isnan(ds[str(vn)].values))==False: 
        # conditions to filter everything BUT CHL and TURB:
        if (str(vn)!='CHL'): #and (str(vn)!='TURB'):
            if str(sn)=='CAPEELIZABETH':
                condition1 = daily_stats['count']<3
                condition2 = time_helper['DT']<timedelta(hours=13) # ideally want times where: 0:17 to 21:17 UTC but will take 18h
            else:
                condition1 = daily_stats['count']<5
                condition2 = time_helper['DT']<timedelta(hours=16) # ideally want times where: 0:17 to 21:17 UTC but will take 
            mask = condition1 | condition2 
            print('mask A '+vn)
        else:
            mask=(daily_stats['count']<2)
            print('mask B '+vn)
            # 2 samples / day min for CHL
        
        # drop rows where mask is True 
        filtered_df = daily_stats.drop(daily_stats[mask].index)
        temptime = pd.to_datetime(filtered_df.index)+pd.Timedelta(hours=12)
        plt.plot(temptime,filtered_df['daily_avg'],color = 'pink',marker='.',linestyle='none',linewidth=3,label ='DailyAvg')  
    
        # check if you dropped the first of last date in the dataset 
        filtered_df['date'] = pd.to_datetime(filtered_df.index) 
        if filtered_df['date'][0]>dt[0].normalize():
            new_row = pd.DataFrame({'date':[dt[0].normalize()],'daily_avg':[np.NaN],'count':[np.NaN]})
            filtered_df = pd.concat([new_row,filtered_df],ignore_index=True)
        if filtered_df['date'][len(filtered_df['date'])-1]<dt[-1].normalize():
            new_row = pd.DataFrame({'date':[dt[-1].normalize()],'daily_avg':[np.NaN],'count':[np.NaN]})
            filtered_df = pd.concat([filtered_df,new_row],ignore_index=True)
    
        # and then resample to fill "empty days" with NaNs   
        filtered_df.set_index('date', inplace=True)
        df_resampled = filtered_df.resample('D').asfreq()
        df_resampled['datetime_utc'] = pd.to_datetime(df_resampled.index)+pd.Timedelta(hours=12)
        plt.plot(df_resampled['datetime_utc'],df_resampled['daily_avg'],color = 'DodgerBlue',marker='.',linestyle='-',linewidth=3,label ='Resampled')  

        ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
        plt.plot(df_resampled['datetime_utc'],df_resampled['count'],color = 'DodgerBlue',marker='.',linestyle='-',linewidth=3,label ='counts')  
    
        if (str(vn)=='CHL'):
            ax1.set_ylim([1,11])
            ax1.set_yticks([1,3,5,7,9,11])
        else:
            if (str(sn)=='CAPEELIZABETH'):
                ax1.set_ylim([2,50])
                ax1.set_yticks([2,4,8,15,25,35,45,50])
            else: 
                ax1.set_ylim([4,15])
                ax1.set_yticks([5,7,9,11,13,15])
       
        #fix the time axis 
        tstart = (df_resampled['datetime_utc'][0] - pd.DateOffset(months=6)).normalize()
        tend = (df_resampled['datetime_utc'][-1] + pd.DateOffset(months=6)).normalize()
        mdates0 = pd.date_range(start=tstart,end=tend, freq='9M')
        date_format = DateFormatter('%d%b%y')
        ax0.set_xlim([mdates0[0],mdates0[-1]])
        ax1.set_xlim([mdates0[0],mdates0[-1]])
        ax0.set_xticks(mdates0)
        ax1.set_xticks(mdates0)
        ax0.xaxis.set_major_formatter(date_format)
        ax1.xaxis.set_major_formatter(date_format)
    else: # no values for the vn at that sn
        ax0.set_title(str(sn)+': sutton et al. 2019 NO DATA')
        ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
        plt.plot(daily_stats.index,daily_stats['count'],color = 'DodgerBlue',marker='.',linestyle='-',linewidth=3,label ='counts')  
    
    ax1.set_ylabel('#obs/daily avg.')
    plt.grid(True)
    
    #if str(vn)=='xCO2_air': sys.exit()
    
    # put the data in a dataset. replace this with try/catch it's ugly
    if ('ds_daily' in locals()) and (np.all(np.isnan(ds[str(vn)].values))==False): #dataset exists and and the variable has values
        ds_daily[str(vn)] = xr.DataArray(df_resampled['daily_avg'].values,dims=('datetime_utc'),
            attrs={'units':ds[str(vn)].attrs['units'], 'long_name':ds[str(vn)].attrs['long_name'], 
            'depth': ds[str(vn)].attrs['depth']})
    elif ('ds_daily' in locals()) and (np.all(np.isnan(ds[str(vn)].values))==True):
        print(str(vn)+' at '+str(sn)+' is all nans; check associated testing plot')
    elif (str(vn)=='SA'):   #initialize new dataset and fill
        coords = {'datetime_utc':('datetime_utc',df_resampled['datetime_utc'])}
        ds_daily = xr.Dataset(coords=coords, 
        attrs={'Station Name':sn_name_dict[sn],'lon':ds.lon,'lat':ds.lat,
        'Depth':'surface ~0.5m', 'Source file':str(fn_in),'data processed': 'Daily Averages @ 12UTC'})
        
        ds_daily[str(vn)] = xr.DataArray(df_resampled['daily_avg'].values,dims=('datetime_utc'),
        attrs={'units':ds[str(vn)].attrs['units'], 'long_name':ds[str(vn)].attrs['long_name'],
        'depth': ds[str(vn)].attrs['depth']})
    
    #if not testing:
    #    ds_daily.to_netcdf(fn_out, unlimited_dims='time')
    #    plt.gcf().tight_layout()
    #    plt.gcf().savefig(fig_out)
        


