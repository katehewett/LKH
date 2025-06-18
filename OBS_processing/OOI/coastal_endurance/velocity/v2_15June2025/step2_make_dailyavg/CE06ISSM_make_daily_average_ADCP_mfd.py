'''
Calc daily avgs to compare with lowpass (and avg) mooring extractions.

pickled files are saved for each oUd
'''

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
from datetime import timedelta
from datetime import datetime
import pickle 
from scipy import stats

import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart()
###############################################################################################################################
###############################################################################################################################
# processed data location
#otype = 'moor' 
moor = 'CE07SHSM'
loco = 'mfd'

Zcenter = np.arange(-29,-4,5)
binedges = Zcenter-2.5
binedges = np.append(binedges,binedges[-1]+5)
###############################################################################################################################
###############################################################################################################################

if loco == 'nsif':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'nsif' 
    fn_in = posixpath.join(in_dir, (moor+'_nsif_ADCP.nc')) 
elif loco == 'mfd':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'mfd' 
    fn_in = posixpath.join(in_dir, (moor+'_mfd_ADCP.nc')) 

out_dir = in_dir / 'daily'

if loco == 'nsif':
    fn_out = posixpath.join(out_dir, (moor+'_nsif_ADCP_DAILY.nc'))
    fig_out_dir =  Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'nsif' / 'plots' / 'daily_avg_processing'
elif loco == 'mfd':
    fn_out = posixpath.join(out_dir, (moor+'_mfd_ADCP_DAILY.nc'))
    fig_out_dir =  Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'mfd' / 'plots' / 'daily_avg_processing'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(fig_out_dir)==False:
    Lfun.make_dir(fig_out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

DTmax = 12

ds = xr.open_dataset(fn_in, decode_times=True)  
NT = np.shape(ds.u)[0]
NZ = np.shape(ds.u)[1]
NZc = np.shape(Zcenter)[0]
u = ds.u.values
v = ds.v.values
z = ds.z.values
##########################################################################
# ADCP were binned at diff depths/deployment and we want in a standard 
# depth, so we're going to put into 15m bins using Zcenter (def'n above)
# (1) grab binnumber as index for np.nanmean
zi = z[0,:]
zstat, bin_edges, binnumber = stats.binned_statistic(zi, zi, statistic='count', bins=binedges)

Uint = np.ones([NT,NZc])*np.nan
Vint = np.ones([NT,NZc])*np.nan

b = binnumber-1
for idx in range(NZc):
    ui = u[:,b==idx]
    vi = v[:,b==idx]
    A = np.nanmean(ui,axis=1)
    B = np.nanmean(vi,axis=1)
    Uint[:,idx] = A
    Vint[:,idx] = B

##########################################################################
# (2) put to df so can bin by day
dt = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df = pd.DataFrame({'ocean_time':dt}) 
df['date'] = df['ocean_time'].dt.date
df['time_diff'] = df['ocean_time'].diff()
df['LT_DTmax'] = df['ocean_time'].diff()< pd.Timedelta(hours=DTmax)
'''
if df['ocean_time'].is_monotonic_increasing == False:
    print('issue with times')
    sys.exit()
else: 
    print('times pass')
'''
##########################################################################
#  U to dataframe 
# TODO: replace ugly loops
for i in range(NZc):
    df2 = df.copy()
    # replace values in time_diff with NaT where column LT_DTmax is False 
    # (that means DT is > DTmax, eg 12hours)
    df2['time_diff']= df2['time_diff'].where(df2['LT_DTmax'],pd.NaT) 

    df2[str(Zcenter[i])] = Uint[:,i]
    df2['N'] = ~np.isnan(df2[str(Zcenter[i])])
    df2['time_diff']= df2['time_diff'].where(df2['N'],pd.NaT) 

    # take daily avg
    daily_stats = df2.groupby('date').agg(daily_avg=(str(Zcenter[i]), 'mean'), count=(str(Zcenter[i]),'count'), coverage = ('time_diff','sum'))

    # replace values where there isn't more than 12 hours of coverage for a daily average
    daily_stats['daily_avg'] = daily_stats['daily_avg'].where(daily_stats['coverage']>timedelta(hours=12),np.nan) 

    if i == 0:
        Udaily = pd.DataFrame({str(Zcenter[i]):daily_stats['daily_avg']})
    else:
        d3 = pd.DataFrame({str(Zcenter[i]):daily_stats['daily_avg']})
        Udaily = pd.concat([Udaily,d3],axis=1)

upkl = posixpath.join(out_dir, (moor+'_Udaily_' + loco + '_ADCP.pkl'))
picklepath = out_dir/upkl
with open(picklepath, 'wb') as fm:
    pickle.dump(Udaily, fm)
    print('Pickled U daily')
    sys.stdout.flush()

##########################################################################
#  V to dataframe 
for i in range(NZc):
    df2 = df.copy()
    # replace values in time_diff with NaT where column LT_DTmax is False 
    # (that means DT is > DTmax, eg 12hours)
    df2['time_diff']= df2['time_diff'].where(df2['LT_DTmax'],pd.NaT) 

    df2[str(Zcenter[i])] = Vint[:,i]
    df2['N'] = ~np.isnan(df2[str(Zcenter[i])])
    df2['time_diff']= df2['time_diff'].where(df2['N'],pd.NaT) 

    # take daily avg
    daily_stats = df2.groupby('date').agg(daily_avg=(str(Zcenter[i]), 'mean'), count=(str(Zcenter[i]),'count'), coverage = ('time_diff','sum'))

    # replace values where there isn't more than 12 hours of coverage for a daily average
    daily_stats['daily_avg'] = daily_stats['daily_avg'].where(daily_stats['coverage']>timedelta(hours=12),np.nan) 

    if i == 0:
        Vdaily = pd.DataFrame({str(Zcenter[i]):daily_stats['daily_avg']})
    else:
        d3 = pd.DataFrame({str(Zcenter[i]):daily_stats['daily_avg']})
        Vdaily = pd.concat([Vdaily,d3],axis=1)

vpkl = posixpath.join(out_dir, (moor+'_Vdaily_' + loco + '_ADCP.pkl'))
picklepath = out_dir/vpkl
with open(picklepath, 'wb') as fm:
    pickle.dump(Vdaily, fm)
    print('Pickled V daily')
    sys.stdout.flush()

####################################################################################################################################################
# replace dt1 and dt2 at start and end so that we can have full years spaced at daily interval
dt1 = datetime(2015,1,1)
dt2 = datetime(2024,12,31)

print('nanfill missing dates ... ')
# resample to fill "empty days" with NaNs
Udaily = Udaily.reset_index()
Vdaily = Vdaily.reset_index()

Udaily.loc[0,'date']=dt1
Udaily.loc[len(Udaily)+1,'date']=dt2

Vdaily.loc[0,'date']=dt1
Vdaily.loc[len(Vdaily)+1,'date']=dt2
####################################################################################################################################################

Udaily['datetimes']=pd.to_datetime(Udaily['date'])
Vdaily['datetimes']=pd.to_datetime(Vdaily['date'])
#Wdaily['datetimes']=pd.to_datetime(Udaily['date'])
#Edaily['datetimes']=pd.to_datetime(Edaily['date'])

Udaily.set_index('datetimes',inplace=True)
Vdaily.set_index('datetimes',inplace=True)
#Wdaily.set_index('datetimes',inplace=True)
#Edaily.set_index('datetimes',inplace=True)

u_resampled = Udaily.resample('D').asfreq()
v_resampled = Vdaily.resample('D').asfreq()
#w_resampled = Wdaily.resample('D').asfreq()
#e_resampled = Edaily.resample('D').asfreq()

u_resampled = u_resampled.drop('date',axis=1)
v_resampled = v_resampled.drop('date',axis=1)
#w_resampled = w_resampled.drop('date',axis=1)
#e_resampled = e_resampled.drop('date',axis=1)

# set up time and then pack for saving in LO friendly format
tdaily = u_resampled.index + timedelta(hours=12)
NT = np.shape(tdaily)[0]
NZ = np.shape(Zcenter)[0]

Uarr = u_resampled.to_numpy()
Varr = v_resampled.to_numpy()
#Warr = w_resampled.to_numpy()
#Earr = e_resampled.to_numpy()

if (np.shape(Varr)[0]==NT) and (np.shape(Varr)[1]==NZ):
    print('shapes ok, putting to dataset') 
else: 
    print('error in shape')
    sys.exit()

Zi = np.tile(Zcenter,(NT,1))

ADCP = xr.Dataset()
ADCP['ocean_time'] = (('ocean_time'), tdaily, {'long_name':'daily average timestamps'})

ADCP['z'] = (('ocean_time','z'), Zi, {'units':'m', 'long_name':'altitude from OOI', 'positive':'up'})

ADCP['u'] = (('ocean_time','z'), Uarr, {'units':'m.s-1', 'long_name': 'Daily avg OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity',
                                      'source_file': fn_in})
ADCP['v'] = (('ocean_time','z'), Varr, {'units':'m.s-1', 'long_name': 'Daily avg OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity',
                                      'source_file': fn_in})
#ADCP['w'] = (('ocean_time','z'), Warr, {'units':'m.s-1', 'long_name': 'Daily avg Upward Sea Water Velocity','positive':'upward',
#                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity',
#                                      'source_file': fn_in})
#ADCP['velprof'] = (('ocean_time','z'), Earr, {'units':'m.s-1', 'long_name': 'Daily avg Error Seawater Velocity detail link broken','positive':'unclear',
#                                      'metadata link':'BROKEN: https://mmisw.org/ont/cf/parameter/VELPROF-EVL',
#                                      'source_file': fn_in})

if loco == 'nsif':
    ADCP['u'].attrs['moored_location'] = 'nsif ~7m below surface'
    ADCP['v'].attrs['moored_location'] = 'nsif ~7m below surface'
    #ADCP['w'].attrs['moored_location'] = 'nsif ~7m below surface'
elif loco == 'mfd':
    if moor == 'CE09OSSM':
        ADCP['u'].attrs['moored_location'] = 'mfd, depth ~540m'
        ADCP['v'].attrs['moored_location'] = 'mfd, depth ~540m'
        #ADCP['w'].attrs['moored_location'] = 'mfd, depth ~540m'
    elif moor == 'CE07SHSM':
        ADCP['u'].attrs['moored_location'] = 'mfd, depth ~87m'
        ADCP['v'].attrs['moored_location'] = 'mfd, depth ~87m'
        #ADCP['w'].attrs['moored_location'] = 'mfd, depth ~87m'
    elif moor == 'CE06ISSM':
        ADCP['u'].attrs['moored_location'] = 'mfd, depth ~29m'
        ADCP['v'].attrs['moored_location'] = 'mfd, depth ~29m'
        #ADCP['w'].attrs['moored_location'] = 'mfd, depth ~29m'

ADCP.to_netcdf(fn_out, unlimited_dims='ocean_time')

print('saved!')
sys.exit()

# PLOTTING
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# map
plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

axu = plt.subplot2grid((4,2), (0,0), colspan=2,rowspan=1)
cpu = axu.pcolormesh(ADCP['ocean_time'],ADCP['z'],ADCP['u'],vmin=-0.5,vmax=0.5,cmap=cm.roma_r)
fig.colorbar(cpu, ax=axu,label = 'daily mean u +east')
axu.set_ylabel('Z [m]')
axu.set_title(moor+ ' ADCP @ mfd')

axv = plt.subplot2grid((4,2), (1,0), colspan=2,rowspan=1)
cpv = axv.pcolormesh(ADCP['ocean_time'],ADCP['z'],ADCP['v'],vmin=-1,vmax=1,cmap=cm.roma_r)
fig.colorbar(cpv, ax=axv,label = 'daily mean v +north')
axv.set_ylabel('Z [m]')

axw = plt.subplot2grid((4,2), (2,0), colspan=2,rowspan=1)
cpw = axw.pcolormesh(ADCP['ocean_time'],ADCP['z'],ADCP['v'],vmin=-0.05,vmax=0.05,cmap=cm.roma_r)
fig.colorbar(cpw, ax = axw, label = 'w +up')
axw.set_ylabel('Z [m]')

axe = plt.subplot2grid((4,2), (3,0), colspan=2,rowspan=1)
cpe = axe.pcolormesh(ADCP['ocean_time'],ADCP['z'],ADCP['velprof'],vmin=-0.1,vmax=0.1,cmap=cm.roma_r)
fig.colorbar(cpe, ax=axe,label = 'daily mean velprof')
axw.set_ylabel('Z [m]')

fig.tight_layout()

'''
# PLOTTING
# map
plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

axu = plt.subplot2grid((4,2), (0,0), colspan=2,rowspan=1)
cpu = axu.pcolormesh(tdaily,z,ub,vmin=-0.5,vmax=0.5,cmap=cm.roma_r)

axv = plt.subplot2grid((4,2), (1,0), colspan=2,rowspan=1)
cpv = axv.pcolormesh(tdaily,z,vb,vmin=-0.5,vmax=0.5,cmap=cm.roma_r)

axw = plt.subplot2grid((4,2), (2,0), colspan=2,rowspan=1)
cpw = axw.pcolormesh(tdaily,z,wb,vmin=-0.5,vmax=0.5,cmap=cm.roma_r)

axe = plt.subplot2grid((4,2), (3,0), colspan=2,rowspan=1)
cpe = axe.pcolormesh(tdaily,z,eb,vmin=-0.5,vmax=0.5,cmap=cm.roma_r)




sys.exit()
#w = ds.w.values
e = ds.velprof.values
z = ds.z.values

A = np.expand_dims(dt,axis=1)
AA = np.tile(A,NZ)

Zi = np.tile(Zcenter,(NT,1))

Uint = np.ones([NT,NZc])*np.nan
Vint = np.ones([NT,NZc])*np.nan



for idx in range(NT):
    ui = u[idx,:]
    vi = v[idx,:]
    zin = z[idx,:]
    zi = zin[~np.isnan(ui)]

    idxS = np.argmin(np.abs(Zcenter - zi[-1]))
    idxB = np.argmin(np.abs(Zcenter - zi[0]))

    unew = np.interp(Zcenter,zi,ui[~np.isnan(ui)])
    Uint[idx,:] = unew

    vnew = np.interp(Zcenter,zi,vi[~np.isnan(vi)])
    Vint[idx,:] = vnew

    idxS = np.argmin(np.abs(Zcenter - zi[-1])) #surface nearest index
    idxB = np.argmin(np.abs(Zcenter - zi[0]))  #bottom nearest index
    if idxS<32: 
        Vint[idx,idxS+1:NZc-1] = np.nan
        Uint[idx,idxS+1:NZc-1] = np.nan
    if idxS == 32:
        Vint[idx,NZc-1] = np.nan
        Uint[idx,NZc-1] = np.nan
    
    if idxB>0: 
        Vint[idx,0:idxB] = np.nan
        Uint[idx,0:idxB] = np.nan
'''