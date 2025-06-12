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
moor = 'CE07SHSM'
#moor = 'CE09OSSM'
#loco = 'mfd'
loco = 'nsif'

thisyr = 2017

if loco == 'nsif':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'nsif' 
    fn_in = posixpath.join(in_dir, (moor+'_nsif_ADCP.nc')) 
elif loco == 'mfd':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'mfd'
    fn_in = posixpath.join(in_dir, (moor+'_mfd_ADCP.nc')) 

out_dir = Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'monthly_stats'

if loco == 'nsif':
    fn_out = posixpath.join(out_dir, (moor+'_nsif_ADCP_monthly'+str(thisyr)+'.nc'))
    #fig_out_dir =  Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'nsif' / 'plots' / 'daily_avg_processing'
elif loco == 'mfd':
    fn_out = posixpath.join(out_dir, (moor+'_mfd_ADCP_monthly'+str(thisyr)+'.nc'))
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

'''
# take whole range and 10*IQR = outlier test
U = u.ravel()
Uclean = U[~np.isnan(U)]
p50 = np.percentile(Uclean,50)
p75 = np.percentile(Uclean,75)
p25 = np.percentile(Uclean,25)
IQR = p75-p25
Uupper = round(p75+10*IQR,1)
Ulower = round(p25-10*IQR,1)

V = v.ravel()
Vclean = V[~np.isnan(U)]
p50 = np.percentile(Vclean,50)
p75 = np.percentile(Vclean,75)
p25 = np.percentile(Vclean,25)
IQR = p75-p25
Vupper = round(p75+10*IQR,1)
Vlower = round(p25-10*IQR,1)

W = w.ravel()
Wclean = W[~np.isnan(W)]
p50 = np.percentile(Wclean,50)
p75 = np.percentile(Wclean,75)
p25 = np.percentile(Wclean,25)
IQR = p75-p25
Wupper = round(p75+10*IQR,1)
Wlower = round(p25-10*IQR,1)
'''

NT = np.shape(ds.u[mdx])[0]
NZ = np.shape(ds.u[mdx])[1]
ot = dt[mdx]
z = ds.z.values[0,:] 
zstring = [str(number) for number in z]

vn_list = list(ds.data_vars)

DTmax = 5
##########################################################################
# df2.iloc[np.where((df2.index.month==10)&(df2.index.day==2|3))]
# Calc Monthly Stats U 
for vn in vn_list:
    df = pd.DataFrame({'ocean_time':ot}) 
    df = df.set_index('ocean_time')
    df[zstring] = ds[vn].values[mdx]
    pmean = df.resample('M').mean()
    #p25 = df.resample('M').quantile(0.25)
    #p75 = df.resample('M').quantile(0.75)
    pstdev = df.resample('M').std()

    condition = ~np.isnan(df[zstring]) #find where missing data, False = nan value in vel array

    df2 = pd.DataFrame({'ocean_time':ot}) 
    #df2['LT_DTmax'] = df2['ocean_time'].diff()< pd.Timedelta(hours=DTmax)
    df2 = df2.set_index('ocean_time')
    for idx in range(NZ):
        df2[str(z[idx])] = df2.index.diff()

    df2 = df2.where(condition, other=pd.Timedelta(hours=0)) # set value to 0 if missing value in vel array 
    condition2 = df2<timedelta(hours=DTmax) # deal with big gaps
    df2 = df2.where(condition2, other=pd.Timedelta(hours=0)) 

    coverage = df2.resample('M').sum()

    stats = {}
    stats['z'] = z
    stats['mean'] = pmean
    #stats['p75'] = p75
    #stats['p25'] = p25
    stats['stdev'] = pstdev
    stats['interval'] = 'monthly'
    stats['location'] = moor + '_ADCP_' +loco
    stats['component'] = vn
    stats['units'] = 'm/s'
    stats['year'] = thisyr
    stats['coverage'] = coverage

    if vn == 'u':
        pkl = posixpath.join(out_dir, (moor+'_Umonthly_' + loco + '_ADCP_'+str(thisyr)+'.pkl'))
    if vn == 'v':
        pkl = posixpath.join(out_dir, (moor+'_Vmonthly_' + loco + '_ADCP_'+str(thisyr)+'.pkl'))
    if vn == 'w':
        pkl = posixpath.join(out_dir, (moor+'_Wmonthly_' + loco + '_ADCP_'+str(thisyr)+'.pkl'))
    elif vn == 'velprof':
        pkl = posixpath.join(out_dir, (moor+'_Emonthly_' + loco + '_ADCP_'+str(thisyr)+'.pkl'))

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