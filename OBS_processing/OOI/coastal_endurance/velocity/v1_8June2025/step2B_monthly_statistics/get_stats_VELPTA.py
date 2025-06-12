'''
Calc monthly stats to compare with lowpass (and avg) mooring extractions.

pickled files are saved for each oUd
'''

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
from datetime import timedelta
import pickle 

import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart()

# processed data location
# TODO add in as args
moor = 'CE06ISSM'
#moor = 'CE07SHSM'
#moor = 'CE09OSSM'
#loco = 'surfacebuoy'
loco = 'nsif'

thisyr = 2017

if loco == 'nsif':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco
    fn_in = posixpath.join(in_dir, (moor+'_nsif_VELPTA.nc')) 
elif loco == 'surfacebuoy':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco
    fn_in = posixpath.join(in_dir, (moor+'_surfacebuoy_VELPTA.nc')) 

out_dir = Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'monthly_stats'

if loco == 'nsif':
    fn_out = posixpath.join(out_dir, (moor+'_nsif_VELPTA_monthly'+str(thisyr)+'.nc'))
    #fig_out_dir =  Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'nsif' / 'plots' / 'daily_avg_processing'
elif loco == 'surfacebuoy':
    fn_out = posixpath.join(out_dir, (moor+'_surfacebuoy_VELPTA_monthly'+str(thisyr)+'.nc'))
    #fig_out_dir =  Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'mfd' / 'plots' / 'daily_avg_processing'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
#if os.path.exists(fig_out_dir)==False:
#    Lfun.make_dir(fig_out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

ds = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
yrs = dt.year
mdx = np.where(yrs==thisyr)
ot = dt[mdx]

NZ = np.shape(ds.u[mdx])[0]
NT = np.shape(ot)[0]

if loco == 'surfacebuoy':
    z = -1
if loco == 'nsif':
    z = -7
#zstring = str(z)

vn_list = list(ds.data_vars)

DTmax = 5

##########################################################################
# df2.iloc[np.where((df2.index.month==10)&(df2.index.day==2|3))]
# Calc Monthly Stats U 
for vn in vn_list:
    df = pd.DataFrame({'ocean_time':ot}) 
    df = df.set_index('ocean_time')
    df['val'] = ds[vn].values[mdx]
    pmean = df.resample('M').mean()
    #p25 = df.resample('M').quantile(0.25)
    #p75 = df.resample('M').quantile(0.75)
    pstdev = df.resample('M').std()

    condition = ~np.isnan(df['val']) #find where missing data, False = nan value in vel array

    df2 = pd.DataFrame({'ocean_time':ot}) 
    #df2['LT_DTmax'] = df2['ocean_time'].diff()< pd.Timedelta(hours=DTmax)
    df2 = df2.set_index('ocean_time')
    df2['time_diff'] = df2.index.diff()

    df2 = df2.where(condition, other=pd.Timedelta(hours=0)) # set value to 0 if missing value in vel array 
    condition2 = df2<timedelta(hours=DTmax) # deal with big gaps
    df2 = df2.where(condition2, other=pd.Timedelta(hours=0)) 

    coverage = df2.resample('M').sum()

    stats = {}
    stats['z'] = str(z) + ' m below surface'
    stats['mean'] = pmean
    #stats['p75'] = p75
    #stats['p25'] = p25
    stats['stdev'] = pstdev
    stats['interval'] = 'monthly'
    stats['location'] = moor + '_VELPTA_' +loco
    stats['component'] = vn
    stats['units'] = 'm/s'
    stats['year'] = thisyr
    stats['coverage'] = coverage

    if vn == 'u':
        pkl = posixpath.join(out_dir, (moor+'_Umonthly_' + loco + '_VELPTA_'+str(thisyr)+'.pkl'))
    if vn == 'v':
        pkl = posixpath.join(out_dir, (moor+'_Vmonthly_' + loco + '_VELPTA_'+str(thisyr)+'.pkl'))
    if vn == 'w':
        pkl = posixpath.join(out_dir, (moor+'_Wmonthly_' + loco + '_VELPTA_'+str(thisyr)+'.pkl'))
    elif vn == 'velprof':
        pkl = posixpath.join(out_dir, (moor+'_Emonthly_' + loco + '_VELPTA_'+str(thisyr)+'.pkl'))

    picklepath = out_dir/pkl
    with open(picklepath, 'wb') as fm:
        pickle.dump(stats, fm)  
        print('Pickled '+vn+' monthly')
        sys.stdout.flush()


'''
##########################################################################
# Calc Monthly Stats V 
df = pd.DataFrame({'ocean_time':ot}) 
df = df.set_index('ocean_time')
df[zstring] = v
pmean = df.resample('M').mean()
p25 = df.resample('M').quantile(0.25)
p75 = df.resample('M').quantile(0.75)

condition = ~np.isnan(df[zstring]) #find where missing data, False = nan value in vel array

df2 = pd.DataFrame({'ocean_time':ot}) 
#df2['LT_DTmax'] = df2['ocean_time'].diff()< pd.Timedelta(hours=DTmax)
df2 = df2.set_index('ocean_time')
for idx in range(NZ):
    df2[str(z[idx])] = df2.index.diff()

df2 = df2.where(condition, other=pd.Timedelta(hours=0))
coverage = df2.resample('M').sum()

stats = {}
stats['z'] = z
stats['mean'] = pmean
stats['p75'] = p75
stats['p25'] = p25
stats['interval'] = 'monthly'
stats['location'] = 'v_'+ moor + '_ADCP_' +loco
stats['units'] = 'm/s'
stats['year'] = thisyr
stats['coverage'] = coverage

vpkl = posixpath.join(out_dir, (moor+'_Vmonthly_' + loco + '_ADCP.pkl'))
picklepath = out_dir/vpkl
with open(picklepath, 'wb') as fm:
    pickle.dump(stats, fm)
    print('Pickled V monthly')
    sys.stdout.flush()

##########################################################################
# Calc Monthly Stats W
df = pd.DataFrame({'ocean_time':ot}) 
df = df.set_index('ocean_time')
df[zstring] = w
pmean = df.resample('M').mean()
p25 = df.resample('M').quantile(0.25)
p75 = df.resample('M').quantile(0.75)

condition = ~np.isnan(df[zstring]) #find where missing data, False = nan value in vel array

df2 = pd.DataFrame({'ocean_time':ot}) 
#df2['LT_DTmax'] = df2['ocean_time'].diff()< pd.Timedelta(hours=DTmax)
df2 = df2.set_index('ocean_time')
for idx in range(NZ):
    df2[str(z[idx])] = df2.index.diff()

df2 = df2.where(condition, other=pd.Timedelta(hours=0))
coverage = df2.resample('M').sum()

stats = {}
stats['z'] = z
stats['mean'] = pmean
stats['p75'] = p75
stats['p25'] = p25
stats['interval'] = 'monthly'
stats['location'] = 'w_'+ moor + '_ADCP_' +loco
stats['units'] = 'm/s'
stats['year'] = thisyr
stats['coverage'] = coverage

wpkl = posixpath.join(out_dir, (moor+'_Wmonthly_' + loco + '_ADCP.pkl'))
picklepath = out_dir/wpkl
with open(picklepath, 'wb') as fm:
    pickle.dump(stats, fm)
    print('Pickled W monthly')
    sys.stdout.flush()
'''