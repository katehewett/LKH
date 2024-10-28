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

Note: the data is NOT 3 hourly, it has lots of diff time steps so take care when 
averaging daily values - need a count and a coverage number 
Need least 12 hours of coverage to make a daily avg for all variables 
We are not taking daily avgs of chla and turbidity here, bc they are only 
sampled sometimes 2x per day etc, and we aren't going to be comparing to LO data 

pH isn't measured it's calculated, and we are not saving that data 

Could output the coverage and counts for the daily data, but for now, we aren't saving those data
just change so save pickled files of df_resampled which would give:     
date/daily_avg/count/coverage/filtered_avg/datetime_utc

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
    'CHABA':'Chaba'
} 
sn_list = list(sn_name_dict.keys())

cdx = 1 
DTmax = 12

sn = sn_list[0]
print(sn)
    
fn_in = posixpath.join(in_dir, (sn +'_QC_CO2test.nc'))
fn_out = posixpath.join(out_dir, (sn + '_daily.nc'))

ds = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(ds.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df = pd.DataFrame({'date':dt.normalize(),'time_utc':dt}) 
df['time_diff'] = df['time_utc'].diff()
df['LT_DTmax'] = df['time_utc'].diff()< pd.Timedelta(hours=DTmax)

if df['time_utc'].is_monotonic_increasing == False:
    print('issue with times')
    sys.exit()
else: 
    print('times pass')

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
     'xCO2_air']
     #'pH']
     #'CHL',
     #'Turbidity']

for vn in var_list2:   
    plt.close('all') 
    fig_out = posixpath.join(fig_out_dir,(sn+'_'+vn+'_DailyAvg.png'))
        
    df[str(vn)] = ds[str(vn)].values

    # where vn == NaN replace time_diff with NaT
    condition = np.isnan(df[str(vn)])
    df.loc[condition==True, 'time_diff'] = pd.Timedelta('NaT')
    
    daily_stats = df.groupby('date').agg(daily_avg=(str(vn), 'mean'), count=(str(vn),'count'), coverage = ('time_diff','sum'))
    
    # initialize 1 plot / deployment so can see easily  
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))
    ax0 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
    plt.plot(dt,ds[vn].values,color = 'black',marker='.',linestyle='none',linewidth=3)
    ax0.set_ylabel(str(vn)+' '+str(ds[str(vn)].attrs['units']))
    ax0.set_title(str(sn)+': sutton et al. 2019 data (greys) with calc daily avgs (blue)')
    plt.grid(True)
    
    plt.plot(daily_stats.index+pd.Timedelta(hours=12),daily_stats['daily_avg'].values,color = 'DodgerBlue',marker='.',linestyle='none',linewidth=3)
    
    # Cape Elizabeth has less variables than the other 2. E.g no DOXY; DO 
    # Go thru exercise if there are values (note: if all = True; vn.values = all nans) 
    if np.all(np.isnan(ds[str(vn)].values))==False: 
        fstdev = np.floor(np.nanstd(ds[str(vn)].values))
        ymin = np.floor(np.nanmin(ds[str(vn)].values))-fstdev
        ymax = np.ceil(np.nanmax(ds[str(vn)].values))+fstdev
        yticks = np.linspace(ymin,ymax,5)
        ax0.set_ylim([ymin,ymax])
        ax0.set_yticks(yticks)
    
    ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
    plt.axhline(y = 12, color = 'r', linestyle = '-') 
    plt.axhline(y = 24, color = 'r', linestyle = '-') 
    plt.plot(daily_stats.index+pd.Timedelta(hours=12),
            (daily_stats['coverage']/datetime.timedelta(hours=1)),
            color = 'DodgerBlue',marker='x',linestyle='none',linewidth=3,label ='coverage')
    yticks1 = np.linspace(0,25,6)
    ax1.set_ylim([0,24])
    ax1.set_yticks(yticks1)
    ax1.set_ylabel('coverage hours')
    plt.grid(True)
    
    if np.all(np.isnan(ds[str(vn)].values))==False:  # if false there are values 
        # where vn == NaN replace time_diff with NaT
        daily_stats['filtered_avg'] = np.copy(daily_stats['daily_avg'])
        condition = daily_stats['coverage']<timedelta(hours=12)
        daily_stats.loc[condition==True, 'filtered_avg'] = np.NaN       

        temptime = pd.to_datetime(daily_stats.index)+pd.Timedelta(hours=12)
        
        ax2 = plt.subplot2grid((3,1),(2,0),colspan=1,rowspan=1)
        plt.plot(temptime,daily_stats['filtered_avg'],color = 'pink',marker='.',linestyle='none',linewidth=3,label ='DailyAvg') 
        
        # resample to fill "empty days" with NaNs 
        daily_stats['date'] = pd.to_datetime(daily_stats.index)
        daily_stats.set_index('date', inplace=True)
        df_resampled = daily_stats.resample('D').asfreq()
        df_resampled['datetime_utc'] = pd.to_datetime(df_resampled.index)+pd.Timedelta(hours=12)
        plt.plot(df_resampled['datetime_utc'],df_resampled['daily_avg'],color = 'DodgerBlue',marker='.',linestyle='-',linewidth=3,label ='Resampled') 
            
    #fix the time axis 
    tstart = (df_resampled['datetime_utc'][0] - pd.DateOffset(months=6)).normalize()
    tend = (df_resampled['datetime_utc'][-1] + pd.DateOffset(months=6)).normalize()
    mdates0 = pd.date_range(start=tstart,end=tend, freq='9M')
    date_format = DateFormatter('%d%b%y')
    ax0.set_xlim([mdates0[0],mdates0[-1]])
    ax1.set_xlim([mdates0[0],mdates0[-1]])
    ax0.set_xticks(mdates0)
    ax1.set_xticks(mdates0)
    ax2.set_xticks(mdates0)
    ax0.xaxis.set_major_formatter(date_format)
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)
    plt.grid(True)
    
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
        

