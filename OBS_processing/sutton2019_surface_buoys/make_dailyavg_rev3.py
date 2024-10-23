'''
AFTER running the following code, 
process_webdata.py
fix_CE_time.py 
QC_spike_test.py
QC_quartile_chem_test.py 

We can now estimate daily averages of mooring data for 3 moornings: 
    Chaba 
    Cape Elizabeth 
    Cape Arago 
 
iput files are saved here: 
/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/CO2_test/

daily files saved here:
/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/daily/

Ran thru this exercise so could compare with lowpass mooring extractions.

'''

testing = False 

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
import datetime 
from datetime import timedelta
import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

Ldir = Lfun.Lstart()

# processed data location
otype = 'moor' 
in_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'CO2_test'
out_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'daily'
fig_out_dir = Ldir['parent'] / 'LKH_output' / 'sutton2019_surface_buoys' / 'plots' / 'daily_avg_processing'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(fig_out_dir)==False:
    Lfun.make_dir(fig_out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

# This is a little lazy, but you need to enter the station name below with the list here:
#sn_name_dict = {
#    'CHABA':'Chaba',
#    'CAPEELIZABETH':'Cape Elizabeth', # less varialbes
#    'CAPEARAGO':'Cape Arago',
#}

sn_name_dict = {
    'CAPEARAGO':'Cape Arago'
} 
sn_list = list(sn_name_dict.keys())

cdx = 1 

sn = sn_list[0]
print(sn)
    
fn_in = posixpath.join(in_dir, (sn +'_QC_CO2test.nc'))
fn_out = posixpath.join(out_dir, (sn + '_daily.nc'))

ds = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(ds.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
var_list = list(ds.keys())
var_list2 = ['SA',
 'SP',
 'IT',
 'CT',
 'SIG0',
 'DO (uM)',
 'DOXY',
 'pCO2_sw',
 'pCO2_air',
 'xCO2_air',
 'pH',
 'CHL',
 'Turbidity']
 
for vn in var_list2:   
    plt.close('all') 
    fig_out = posixpath.join(fig_out_dir,(sn+'_'+vn+'_DailyAvg.png'))
    
    # this is a lot of excessive steps, but is to make prettier plots and avoid the 
    # dreaded ugly line; I wanted to seperate deployment chunks
    df = pd.DataFrame({'date':dt.normalize(),'time_utc':dt,str(vn):ds[vn].values})
    df['time_diff'] = df['time_utc'].diff()
    # because some diffs are 2 hr 59 min; 3 hour 1 min; etc
    # and there are some short dt's e.g. 00:30 min 
    # use 15 min thershold to find all the time deltas under 3 hours 15 mins 
    # Cape Elizabeth changes again... hard code to fix for now. 4 hour DTs in CE  
    if str(sn)=='CAPEELIZABETH':
        df['LT 4 hour'] = df['time_diff']<pd.Timedelta('4 hours 15 minutes')
        x = df['LT 4 hour']
    else: 
        df['LT 3 hour'] = df['time_diff']<pd.Timedelta('3 hours 15 minutes')
        x = df['LT 3 hour']
    
    # find start & stop indicies of all continious data        
    s = pd.Series(x)
    grp = s.eq(False).cumsum()
    arr = grp.loc[s.eq(True)].groupby(grp).apply(lambda x: [x.index.min()-1,x.index.max()])
    a = np.array(arr) # its an array of start and stop indicies
                      # a[0] = index of first chunk of data: (start,end) aka (a[0][0],a[0][1]) 
    
    # initialize 1 plot / deployment so can see easily  
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))
    ax0 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
    plt.plot(dt,ds[vn].values,color = 'black',marker='o',linestyle='none',linewidth=3)
    ax0.set_ylabel(str(vn)+' '+str(ds[str(vn)].attrs['units']))
    ax0.set_title(str(sn)+': sutton et al. 2019 data (greys) with calc daily avgs (blue)')
    plt.grid(True)
    
    # Cape Elizabeth has less variables than the other 2. E.g no DOXY; DO 
    # Go thru exercise if there are values (note: if all = True; vn.values = all nans) 
    if np.all(np.isnan(ds[str(vn)].values))==False: 
        fstdev = np.floor(np.nanstd(ds[str(vn)].values))
        ymin = np.floor(np.nanmin(ds[str(vn)].values))-fstdev
        ymax = np.ceil(np.nanmax(ds[str(vn)].values))+fstdev
        yticks = np.linspace(ymin,ymax,5)
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(yticks)

        # step thru each chunks of continious data from the web
        for idx in range(0,np.shape(a)[0]):
            itime = dt[a[idx][0]:a[idx][-1]]
            ivar = ds[vn].values[a[idx][0]:a[idx][-1]]
            plt.plot(itime,ivar,color = 'LightGrey',marker='.',linestyle='-',linewidth=3,label ='SA')
        
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
    if ('ds_daily' in locals()) and (np.all(np.isnan(ds[str(vn)].values))==False): 
        #dataset exists and and the variable has values
        ds_daily[str(vn)] = xr.DataArray(df_resampled['daily_avg'].values,dims=('datetime_utc'),
            attrs=ds[str(vn)].attrs)
            
    elif ('ds_daily' in locals()) and (np.all(np.isnan(ds[str(vn)].values))==True):
        print(str(vn)+' at '+str(sn)+' is all nans; check associated testing plot')
        
    elif (str(vn)=='SA'):   #first loop and need to initialize new dataset and fill
        coords = {'datetime_utc':('datetime_utc',df_resampled['datetime_utc'])}
        ds_daily = xr.Dataset(coords=coords, 
        attrs=ds.attrs)     
        ds.attrs['Source file'] = str(fn_in)
        ds.attrs['data processed'] = 'Daily Averages @ 12UTC'
        
        ds_daily[str(vn)] = xr.DataArray(df_resampled['daily_avg'].values,dims=('datetime_utc'),
        attrs=ds[str(vn)].attrs)
    
    if not testing:
        ds_daily.to_netcdf(fn_out, unlimited_dims='time')
        plt.gcf().tight_layout()
        plt.gcf().savefig(fig_out)
        


