'''
Daily averages were calculated in step1 
this script just puts in LO datetimes and puts all nan rows where there is a gap 

TODO make to moor and loco to args

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
# TODO make to args
#otype = 'moor' 
#moor = 'CE09OSSM'
#moor = 'CE07SHSM'
moor = 'CE06ISSM'
#loco = 'nsif'
loco = 'mfd'

###############################################################################################################################
###############################################################################################################################

if loco == 'nsif':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'nsif' 
    fn_in = posixpath.join(in_dir, (moor+'_nsif_ADCP_rev1.nc')) 
elif loco == 'mfd':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'mfd' 
    fn_in = posixpath.join(in_dir, (moor+'_mfd_ADCP_rev1.nc')) 

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

ds = xr.open_dataset(fn_in, decode_times=True)  
NT = np.shape(ds.u)[0]
NZ = np.shape(ds.u)[1]
Zcenter = ds.z.values[1,:]
##########################################################################
# (1) put to df so can fill missing days 
dt = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df = pd.DataFrame({'date':dt}) 

Udaily = df.copy()
Vdaily = df.copy()
Edaily = df.copy() 

for idx in range(0,NZ):
    Udaily[str(Zcenter[idx])] = ds.u.values[:,idx]
    Vdaily[str(Zcenter[idx])] = ds.v.values[:,idx]
    Edaily[str(Zcenter[idx])] = ds.e.values[:,idx]

####################################################################################################################################################
# (1) enter 1 jan and 12 dec dates if necessary so we have full years 
t1 = df['date'].iloc[0]
t2 = df['date'].iloc[NT-1]

print('nanfill missing dates ... ')
# resample to fill "empty days" with NaNs
if (t1.year == 2015) and (t2 != datetime(2015,1,1)):
    dt1 = datetime(2015,1,1)
    Udaily.loc[len(Udaily)+1,'date']=dt1
    Vdaily.loc[len(Vdaily)+1,'date']=dt1

if (t1.year == 2014) and (t2 != datetime(2014,1,1)):
    dt1 = datetime(2014,1,1)
    Udaily.loc[len(Udaily)+1,'date']=dt1
    Vdaily.loc[len(Vdaily)+1,'date']=dt1

if t2.year == 2024 and t2 != datetime(2024,12,31):
    dt2 = datetime(2024,12,31)
    Udaily.loc[len(Udaily)+1,'date']=dt2
    Vdaily.loc[len(Vdaily)+1,'date']=dt2

Udaily.sort_values(by='date', ascending=True, inplace = True)
Vdaily.sort_values(by='date', ascending=True, inplace = True)
Udaily.reset_index(drop=True, inplace = True)
Vdaily.reset_index(drop=True, inplace = True)

# (2) resample so we gap fill nans for missing dates 
Udaily.set_index('date',inplace=True)
Vdaily.set_index('date',inplace=True)

u_resampled = Udaily.resample('D').asfreq()
v_resampled = Vdaily.resample('D').asfreq()

####################################################################################################################################################
# set up time and then pack for saving in LO friendly format
tdaily = u_resampled.index + timedelta(hours=12)
NT = np.shape(tdaily)[0]
NZ = np.shape(Zcenter)[0]

Uarr = u_resampled.to_numpy()
Varr = v_resampled.to_numpy()
Earr = v_resampled.to_numpy()

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
ADCP['e'] = (('ocean_time','z'), Earr, {'units':'m.s-1', 'long_name': 'Daily avg Error Seawater Velocity detail link broken','positive':'unclear',
                                      'metadata link':'BROKEN: https://mmisw.org/ont/cf/parameter/VELPROF-EVL',
                                      'source_file': fn_in})

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

print('saved! ' + moor + ' ADCP at: ' +loco )

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