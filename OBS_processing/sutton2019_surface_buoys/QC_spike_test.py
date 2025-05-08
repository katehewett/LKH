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
(2b) Maybe towards end of some deployments a cond. sensor may foul (check if there's a cast) TODO
(3) It looks like, but need to verify, that the data presented passed a gross range test 
or equiv QC, but not others? Need to verify?
(4) The time is not in order and there are duplicates in Cape Elizabeth (run fix_CE_time.py)

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
out_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'spike_test'
fig_out_dir = Ldir['parent'] / 'LKH_output' / 'sutton2019_surface_buoys' / 'plots' / 'spike_test_results'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(fig_out_dir)==False:
    Lfun.make_dir(fig_out_dir, clean = False)
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

if str(sn)=='CAPEELIZABETH':
    fn_in = posixpath.join(in_dir, (sn +'_sorted.nc')) # had to fix times and remove dups in CE 
else: 
    fn_in = posixpath.join(in_dir, (sn +'.nc'))
    
fn_out = posixpath.join(out_dir, (sn + '_QC_spiketest_1.5IQR.nc'))

ds = xr.open_dataset(fn_in, decode_times=True)  

'''
Step 1: Identify deployment chunks using DTmax = 12 hours
        Then step thru spike test 
'''
# DTmax = 15 enter as line argument later
DTmax = 15 

dt = pd.to_datetime(ds.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df = pd.DataFrame({'date':dt.normalize(),'time_utc':dt}) 
df['time_diff'] = df['time_utc'].diff()
df['LT_DTmax'] = df['time_utc'].diff()< pd.Timedelta(hours=DTmax)

if df['time_utc'].is_monotonic_increasing == False:
    print('issue with times')
    sys.exit()
  
# find start & stop indicies of all continious data        
s = pd.Series(df['LT_DTmax'])
grp = s.eq(False).cumsum()
arr = grp.loc[s.eq(True)].groupby(grp).apply(lambda x: [x.index.min()-1,x.index.max()])
a = np.array(arr) # its an array of start and stop indicies
                  # a[0] = index of first chunk of data: (start,end) aka (a[0][0],a[0][1]) 

if str(sn) == 'CAPEELIZABETH': #no oxygen
    var_list1 = ['SP','IT']
else:
    var_list1 = ['SP','IT','DOXY']  
                                         
var_list2 = list(ds.keys())

window_size = 30 # for rolling stats 
min_samples = 10 
                   
for vn in var_list1: 
    # Cape Elizabeth has less variables than the other 2. E.g no DOXY; DO 
    # Go thru exercise if there are values (note: if all = True; vn.values = all nans) 
    if np.all(np.isnan(ds[str(vn)].values))==False:
    
        for idx in range(0,np.shape(a)[0]): # step thru each chunks of continious data from the web
            #seperate chunks:
            itime = dt[a[idx][0]:a[idx][-1]]
            ivar = pd.Series(ds[vn].values[a[idx][0]:a[idx][-1]])
            
            fig_out = posixpath.join(fig_out_dir,(str(sn) + '_'+str(vn)+'_chunk'+str(idx)+'.png'))
        
            # Adopted from Argo QC mannuals (sthresh for SP IT DOXY lt 500m depth) 
            # fail is just sthreshx2 (we're only taking passing data, so doen'st really matter what fail is)
            if str(vn)=='SP': 
                sthresh = 0.9
                fail_thresh = 1.8
            elif str(vn) == 'IT':
                sthresh = 6
                fail_thresh = 12  
            elif str(vn) == 'DOXY':
                sthresh = 50
                fail_thresh = 100
            
            qc_results = qartod.spike_test(inp=ivar,suspect_threshold=sthresh,fail_threshold=fail_thresh,)
            '''  GOOD = 1
                 UNKNOWN = 2
                 SUSPECT = 3
                 FAIL = 4
                 MISSING = 9 '''
            
            df3 = pd.DataFrame({'time_utc':itime,'var':ivar,'QC':qc_results}) 
            df3[str(vn + '_spike1')] = np.nan
            df3['GOOD'] = qc_results==1
            df3['GOOD'].astype(int)
            df3.loc[(qc_results==1), str(vn + '_spike1')] = ivar[qc_results==1]
            
            # Calc outliers, after despiked; like: Documents/LKH/tests/tests_stats/testing_outliers.py 
            despiked = pd.Series(ivar)
            rw = despiked.rolling(window=window_size,min_periods=min_samples,center=True) 
            p25 = rw.quantile(0.25)
            p75 = rw.quantile(0.75)
            iqr = p75 - p25
            lower_bound = p25 - 1.5 * iqr
            upper_bound = p75 + 1.5 * iqr

            outliers = (ivar < lower_bound) | (ivar > upper_bound)
            outlier_vals = outliers.astype(int)*ivar
            outlier_vals.replace(0, np.nan, inplace=True)
            
            # save, so not deleted next time thru loop
            if str(vn)=='SP': 
                if str(vn+'_spike') in locals(): # it exists.
                    SP_time = np.append(SP_time,itime)
                    SP_spike = np.append(SP_spike, qc_results) 
                    SP_outliers = np.append(SP_outliers, outliers)
                else: # doesn't exist 
                    SP_time = itime
                    SP_spike = qc_results
                    SP_outliers = outliers
            elif str(vn) == 'IT':
                if str(vn+'_spike') in locals(): # myVar exists.
                    IT_time = np.append(IT_time,itime)
                    IT_spike = np.append(IT_spike, qc_results) 
                    IT_outliers = np.append(IT_outliers, outliers)
                else: # doesn't exist 
                    IT_time = itime
                    IT_spike = qc_results
                    IT_outliers = outliers
            elif str(vn) == 'DOXY':
                if str(vn+'_spike') in locals(): # myVar exists.
                    DOXY_time = np.append(DOXY_time,itime)
                    DOXY_spike = np.append(DOXY_spike, qc_results) 
                    DOXY_outliers = np.append(DOXY_outliers, outliers)   
                else:
                    DOXY_time = itime
                    DOXY_spike = qc_results
                    DOXY_outliers = outliers   
            
            # plot checks
            plt.close('all')
            fs=12
            plt.rc('font', size=fs)
            fig = plt.figure(figsize=(16,8))
            ax0 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
            ax0.fill_between(itime, lower_bound, upper_bound,color = 'dodgerblue',alpha=0.2)
            #plt.plot(dt,ds[vn].values,color = 'black',marker='.',linestyle='-',linewidth=1)
            ax0.set_ylabel(str(vn)+' '+str(ds[str(vn)].attrs['units']))
            ax0.set_title(str(sn)+': sutton et al. 2019 data (imported) chunk: ' + str(idx))
            plt.grid(True)
            
            plt.plot(itime,ivar,color = 'black',marker='.',linestyle='-',linewidth=1)
            plt.plot(itime,outlier_vals,color = 'red',marker='x',linestyle='-',linewidth=1)
            
            if np.all(np.isnan(ivar))==False:
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
            plt.plot(itime,df3[str(vn + '_spike1')],color = 'blue',marker='.',linestyle='none',linewidth=1)
            plt.plot(itime,outlier_vals,color = 'red',marker='x',linestyle='-',linewidth=1)
            
            if np.all(np.isnan(ivar))==False:    
                ax2.set_ylim([ymin,ymax])
                ax2.set_yticks(yticks)
                ax2.set_xlim([itime[0],itime[-1]])
            
            plt.gcf().tight_layout()
            plt.gcf().savefig(fig_out)

# put the qc data into the dataframe. 
if str(sn) == 'CAPEELIZABETH':
    if np.all(SP_time==IT_time): 
        print('times match')
    else: 
        print('error with times')
        sys.exit()   
else:
    if np.all(SP_time==IT_time) and np.all(DOXY_time==IT_time):
        print('times match')
    else:
        print('error with times')
        sys.exit()
    
# initialize
df['SP_spike'] = np.nan
df['SP_outliers'] = np.nan
df['IT_spike'] = np.nan
df['IT_outliers'] = np.nan
if str(sn) != 'CAPEELIZABETH':
    df['DOXY_spike'] = np.nan
    df['DOXY_outliers'] = np.nan

dates_to_match = pd.to_datetime(SP_time)
df.loc[df['time_utc'].isin(dates_to_match),'SP_spike'] = SP_spike
df.loc[df['time_utc'].isin(dates_to_match),'SP_outliers'] = SP_outliers
df.loc[df['time_utc'].isin(dates_to_match),'IT_spike'] = IT_spike
df.loc[df['time_utc'].isin(dates_to_match),'IT_outliers'] = IT_outliers
if str(sn) != 'CAPEELIZABETH':
    df.loc[df['time_utc'].isin(dates_to_match),'DOXY_spike'] = DOXY_spike
    df.loc[df['time_utc'].isin(dates_to_match),'DOXY_outliers'] = DOXY_outliers
            
# if spike qc = 1 and is NOT an outlier, then will pass
# if TS flagged, then flag other vars
condition1 = (df['SP_spike'] == 1) | (df['SP_spike'] == 9)
condition2 = (df['IT_spike'] == 1) | (df['IT_spike'] == 9)
condition3 = df['SP_outliers'] == False
condition4 = df['IT_outliers'] == False
combined_condition = condition1 & condition2 & condition3 & condition4

if str(sn) != 'CAPEELIZABETH':
    condition5 = df['DOXY_outliers'] == False
    condition6 = (df['DOXY_spike'] == 1) | (df['DOXY_spike'] == 9)
    DO_condition = condition5 & condition6 
 
# make a filtered dataset: 
# if T | S fail -> var_list3 fail 
ds2 = ds.copy() 
for vn in var_list2:
    nan_indicies = np.isnan(ds2[vn].values)
    ds2[vn] = ds2[vn]*combined_condition # Filter with combined_condition QC: True pass; False fail 
    ds2[vn].values[nan_indicies] = np.nan  
    ds2[vn].values[combined_condition==False] = np.nan # want nans not zeros 

# additional QC for O2
if str(sn) != 'CAPEELIZABETH':
    # Do DO uM and DOXY
    nan_indicies = np.isnan(ds2['DOXY'].values)
    ds2['DOXY'] = ds2['DOXY']*DO_condition # Filter with DO_condition QC: True pass; False fail 
    ds2['DOXY'].values[nan_indicies] = np.nan  
    ds2['DOXY'].values[DO_condition==False] = np.nan # want nans not zeros 
  
    nan_indicies = np.isnan(ds2['DO (uM)'].values)
    ds2['DO (uM)'] = ds2['DO (uM)']*DO_condition # Filter with DO_condition QC: True pass; False fail 
    ds2['DO (uM)'].values[nan_indicies] = np.nan  
    ds2['DO (uM)'].values[DO_condition==False] = np.nan # want nans not zeros 
  
ds2['combined_condition'] = xr.DataArray(combined_condition, dims=('time'),
    attrs={'units':'bool', 'long_name':'combined_QC results T S: True pass; False fail'})

# make sure to save attrs 
for vn in var_list2:
    ds2[vn].attrs = ds[vn].attrs 
    
if str(sn) != 'CAPEELIZABETH':
    ds2['DO_condition'] = xr.DataArray(combined_condition, dims=('time'),
    attrs={'units':'bool', 'long_name':'DO QC results: True pass; False fail'})
    
ds2.attrs['Source file'] = fn_in
ds2.attrs['QC'] = 'TSDOXY QC_spike_test and quartiles'

ds2.to_netcdf(fn_out, unlimited_dims='time')
