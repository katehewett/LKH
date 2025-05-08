'''
This code is to run basic QC on NOAA data products for 3 moornings: 
    Chaba 
    Cape Elizabeth 
    Cape Arago 

(see Sutton et al 2014 + 2019) 

This script runs after QC_spike_Test.py input files here: 

This looks at pCO2 data 


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
in_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'spike_test'
out_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'CO2_test'
fig_out_dir = Ldir['parent'] / 'LKH_output' / 'sutton2019_surface_buoys' / 'plots' / 'CO2_test_results'

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
   'CAPEELIZABETH':'Cape Elizabeth'
} 
sn_list = list(sn_name_dict.keys())

#cdx = 1 

sn = sn_list[0]
print(sn)


fn_in = posixpath.join(in_dir, (sn + '_QC_spiketest_1.5IQR.nc'))
mIQR = 1.5
fn_out = posixpath.join(out_dir, (sn + '_QC_CO2test.nc'))

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

var_list1 = ['pCO2_sw','pCO2_air','xCO2_air','pH']                                        
#var_list2 = list(ds.keys())

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
            
            # Calc outliers, after despiked; like: Documents/LKH/tests/tests_stats/testing_outliers.py 
            despiked = pd.Series(ivar)
            rw = despiked.rolling(window=window_size,min_periods=min_samples,center=True) 
            p25 = rw.quantile(0.25)
            p75 = rw.quantile(0.75)
            iqr = p75 - p25
            lower_bound = p25 - mIQR * iqr
            upper_bound = p75 + mIQR * iqr

            outliers = (ivar < lower_bound) | (ivar > upper_bound)
            outlier_vals = outliers.astype(int)*ivar
            outlier_vals.replace(0, np.nan, inplace=True)
            
            # save, so not deleted next time thru loop
            if str(vn)=='pCO2_sw': 
                if str(vn+'_outliers') in locals(): # it exists.
                    pCO2_sw_time = np.append(pCO2_sw_time,itime)
                    pCO2_sw_outliers = np.append(pCO2_sw_outliers, outliers)
                else: # doesn't exist 
                    pCO2_sw_time = itime
                    pCO2_sw_outliers = outliers
            elif str(vn) == 'pCO2_air':
                if str(vn+'_outliers') in locals(): # myVar exists.
                    pCO2_air_time = np.append(pCO2_air_time,itime)
                    pCO2_air_outliers = np.append(pCO2_air_outliers, outliers)
                else: # doesn't exist 
                    pCO2_air_time = itime
                    pCO2_air_outliers = outliers
            elif str(vn) == 'xCO2_air':
                if str(vn+'_outliers') in locals(): # myVar exists.
                    xCO2_air_time = np.append(xCO2_air_time,itime) 
                    xCO2_air_outliers = np.append(xCO2_air_outliers, outliers)   
                else:
                    xCO2_air_time = itime
                    xCO2_air_outliers = outliers   
            elif str(vn) == 'pH':
                if str(vn+'_outliers') in locals(): # myVar exists.
                    pH_time = np.append(pH_time,itime)
                    pH_outliers = np.append(pH_outliers, outliers)   
                else:
                    pH_time = itime
                    pH_outliers = outliers
                    
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
            
            plt.gcf().tight_layout()
            plt.gcf().savefig(fig_out)

# NOTE pH isn't measured it's calculated just running thru w/ it 
# var_list1 = ['pCO2_sw','pCO2_air','xCO2_air','pH']

# put the qc data into the dataframe. 

if np.all(pCO2_air_time==xCO2_air_time) and np.all(pCO2_sw_time==pCO2_air_time) and np.all(pCO2_air_time==pH_time):
    print('times match')
else:
    print('error with times')
    sys.exit()

# initialize
df['pCO2_sw_outliers'] = np.nan
df['pCO2_air_outliers'] = np.nan
df['xCO2_air_outliers'] = np.nan
df['pH_outliers'] = np.nan

dates_to_match = pd.to_datetime(pCO2_sw_time)
df.loc[df['time_utc'].isin(dates_to_match),'pCO2_sw_outliers'] = pCO2_sw_outliers
df.loc[df['time_utc'].isin(dates_to_match),'pCO2_air_outliers'] = pCO2_air_outliers
df.loc[df['time_utc'].isin(dates_to_match),'xCO2_air_outliers'] = xCO2_air_outliers
df.loc[df['time_utc'].isin(dates_to_match),'pH_outliers'] = pH_outliers
           
# if False not an outlier, so if == False, makes passing value = 1 (a little weird. could fix this if do again for more moorings)
condition1 = df['pCO2_sw_outliers'] == False   
condition2 = df['pCO2_air_outliers'] == False
condition3 = df['xCO2_air_outliers'] == False
condition4 = df['pH_outliers'] == False
    
# make a filtered dataset
ds2 = ds.copy() 

nan_indicies = np.isnan(ds2['pCO2_sw'].values)
ds2['pCO2_sw'] = ds2['pCO2_sw']*condition1           # Filter with condition QC: True pass; False fail 
ds2['pCO2_sw'].values[nan_indicies] = np.nan  
ds2['pCO2_sw'].values[condition1==False] = np.nan    # want nans not zeros 

nan_indicies = np.isnan(ds2['pCO2_air'].values)
ds2['pCO2_air'] = ds2['pCO2_air']*condition2         # Filter with condition QC: True pass; False fail 
ds2['pCO2_air'].values[nan_indicies] = np.nan  
ds2['pCO2_air'].values[condition2==False] = np.nan   # want nans not zeros 

nan_indicies = np.isnan(ds2['xCO2_air'].values)
ds2['xCO2_air'] = ds2['xCO2_air']*condition3         # Filter with condition QC: True pass; False fail 
ds2['xCO2_air'].values[nan_indicies] = np.nan  
ds2['xCO2_air'].values[condition3==False] = np.nan   # want nans not zeros 

nan_indicies = np.isnan(ds2['pH'].values)
ds2['pH'] = ds2['pH']*condition4                     # Filter with condition QC: True pass; False fail 
ds2['pH'].values[nan_indicies] = np.nan  
ds2['pH'].values[condition4==False] = np.nan         # want nans not zeros 
  
ds2['pCO2_sw_outliers'] = xr.DataArray(condition1, dims=('time'),
    attrs={'units':'bool', 'long_name':'outlier results: True data pass; False data fail'})
ds2['pCO2_air_outliers'] = xr.DataArray(condition2, dims=('time'),
    attrs={'units':'bool', 'long_name':'outlier results: True data pass; False data fail'})
ds2['xCO2_air_outliers'] = xr.DataArray(condition3, dims=('time'),
    attrs={'units':'bool', 'long_name':'outlier results: True data pass; False data fail'})
ds2['pH_outliers'] = xr.DataArray(condition4, dims=('time'),
    attrs={'units':'bool', 'long_name':'outlier results: True data pass; False data fail'})

ds2.attrs['Source file'] = fn_in
ds2.attrs['QC'] = 'quartiles w/ 30 sample window, outlier = '+str(mIQR)+'IQR'

# make sure to save attrs 
for vn in var_list1:
    ds2[vn].attrs = ds[vn].attrs 

ds2.to_netcdf(fn_out, unlimited_dims='time')
