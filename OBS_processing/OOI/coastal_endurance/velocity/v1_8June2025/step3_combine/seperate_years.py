'''
plot whats there . 
'''

import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta

from lo_tools import Lfun

import matplotlib.pyplot as plt

#TODO I don't understand the error in making int(jj[0]), suppressing error b/c will be a problem when update python but not rn 
import warnings
warnings.filterwarnings("ignore")

Ldir = Lfun.Lstart()

moor = 'CE09OSSM'

yr_list = [year for year in range(2015, 2025)]
numyrs = len(yr_list)

#out_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco / 'daily'

mfd = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/mfd/daily/'+moor+'_mfd_ADCP_DAILY.nc', decode_times=True)     

if (moor == 'CE09OSSM') | (moor == 'CE07SHSM'):
    nsif = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/nsif/daily/'+moor+'_nsif_ADCP_DAILY.nc')

vel7 = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/nsif/daily/'+moor+'_nsif_VELPTA_DAILY.nc')
vel1 = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/surfacebuoy/daily/'+moor+'_surfacebuoy_VELPTA_DAILY.nc')

'''
t1V1 = VEL1.ocean_time[0]
t2V1 = VEL1.ocean_time[-1]

t1V7 = VEL7.ocean_time[0]
t2V7 = VEL7.ocean_time[-1]

t1mfd = mfd.ocean_time[0]
t2mfd = mfd.ocean_time[-1]

t1nsif = nsif.ocean_time[0]
t2nsif = nsif.ocean_time[-1]

print('V1 range: ' + str(t1V1.values) + ' ' + str(t2V1.values))
print('V7 range: ' + str(t1V7.values) + ' ' + str(t2V7.values))
print('mfd range: ' + str(t1mfd.values) + ' ' + str(t2mfd.values))
print('nsif range: ' + str(t1nsif.values) + ' ' + str(t2nsif.values))

plt.plot(mfd.u.values[10,:],mfd.z.values)
plt.plot(nsif.u.values[10,:],nsif.z.values)
'''

mfd_dt = pd.to_datetime(mfd.ocean_time.values)
vel7_dt = pd.to_datetime(vel7.ocean_time.values)
vel1_dt = pd.to_datetime(vel1.ocean_time.values)
nsif_dt = pd.to_datetime(nsif.ocean_time.values)

mfd_yrs = mfd_dt.year
vel7_yrs = vel7_dt.year
vel1_yrs = vel1_dt.year
nsif_dt = nsif_dt.year

vn_list = list(mfd.data_vars)
vn_list2 = list(vel7.data_vars)

NZm = np.shape(mfd.z)[0]

for ydx in range(0,numyrs): 
    thisyr = yr_list[ydx]

    #mfd
    mdx = np.where(mfd_yrs==thisyr)
    print(str(thisyr) + ': mfd ' + str(np.shape(mdx)[1]))

    if np.shape(mdx)[1] < 365:
        dt = mfd_dt[mdx]
        t1 = dt[0]
        t2 = dt[-1]

        start_date = str(thisyr)+'-01-01'
        end_date = str(thisyr)+'-12-31'
        date_array = pd.date_range(start=start_date, end=end_date, freq='D')
        dti = date_array + timedelta(hours=12)

        if dt[-1] == dti[-1]: 
            j = np.where(t1==dti)
            print(str(thisyr) + ' missing start of year')
        elif dt[0] != dti[0]:
            j = np.where(t2==dti)
            print(str(thisyr) + ' missing end of year')

        jj = int(j[0]) # why does this make an error for future in python ?? 
        missingtime = dti[:jj]
        NM = np.shape(missingtime)[0]
        nanarray = np.ones([NM,NZm])*np.nan

        if dt[-1] == dti[-1]: 
            dtj = np.concatenate((missingtime, dt), axis=None)
        elif dt[0] != dti[0]:
            dtj = np.concatenate((dt,missingtime), axis=None)
        if np.all(dtj==dti):
            print('times fixed')
        else:
            print('time error')
            sys.exit()

        u = np.concatenate((nanarray,mfd.u.values[mdx,:].squeeze()), axis=0)
        v = np.concatenate((nanarray,mfd.v.values[mdx,:].squeeze()), axis=0)
        w = np.concatenate((nanarray,mfd.w.values[mdx,:].squeeze()), axis=0)
        e = np.concatenate((nanarray,mfd.e.values[mdx,:].squeeze()), axis=0)

        ds = xr.Dataset()
        ds['ocean_time'] = (('ocean_time'), dtj, {'long_name':'daily average timestamps'})
        ds['z'] = (('z'), z, {'units':'m', 'long_name':'altitude from OOI', 'positive':'up'})
        ds['u'] = (('ocean_time','z'), u, {'units':'m.s-1', 'long_name': 'Daily avg OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'}),
        ds['v'] = (('ocean_time','z'), v, {'units':'m.s-1', 'long_name': 'Daily avg OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity'})
        ds['w'] = (('ocean_time','z'), w, {'units':'m.s-1', 'long_name': 'Daily avg Upward Sea Water Velocity','positive':'upward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity'})
        ds['velprof'] = (('ocean_time','z'), e, {'units':'m.s-1', 'long_name': 'Daily avg Error Seawater Velocity detail link broken','positive':'unclear',
                                      'metadata link':'BROKEN: https://mmisw.org/ont/cf/parameter/VELPROF-EVL'})
            



        sys.exit()


'''
# add any missing attributes
ds = xr.open_dataset(moor_fn)
ds0 = xr.open_dataset(fn_list[0])
vn_list = list(ds0.data_vars)
for vn in vn_list:
    try:
        long_name = ds0[vn].attrs['long_name']
        ds[vn].attrs['long_name'] = long_name
    except KeyError:
        pass
    try:
        units = ds0[vn].attrs['units']
        ds[vn].attrs['units'] = units
    except KeyError:
        pass
ds.close()
ds0.close()
'''   
