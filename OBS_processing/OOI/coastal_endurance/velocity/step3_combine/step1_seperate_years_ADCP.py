'''
seperate years and save in format to match lo and other instruments 
ADCP's 

There is some 2014(CE06ISSM) and 2025(all) data, but omitting for now so can do monthly comparisons

'''

import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta
import posixpath

from lo_tools import Lfun

import matplotlib.pyplot as plt

#TODO I don't understand the error in making int(jj[0]), suppressing error b/c will be a problem when update python but not rn 
import warnings
warnings.filterwarnings("ignore")

Ldir = Lfun.Lstart()

#moor = 'CE09OSSM'
moor = 'CE07SHSM'
#moor = 'CE06ISSM'
loco = 'nsif'
#loco= 'mfd'


yr_list = [year for year in range(2015, 2025)]
numyrs = len(yr_list)

if loco == 'mfd':
    fn_in = '/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/'+loco+'/daily/'+moor+'_mfd_ADCP_DAILY.nc'       
elif loco == 'nsif':
    fn_in = '/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/'+loco+'/daily/'+moor+'_nsif_ADCP_DAILY.nc'

out_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco / 'daily' / 'by_year'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)

ds = xr.open_dataset(fn_in, decode_times=True) 
ot = pd.to_datetime(ds.ocean_time.values)
yrs = ot.year
vn_list = list(ds.data_vars)

NZ = np.shape(ds.z)[0]

for ydx in range(0,numyrs): 
    thisyr = yr_list[ydx]

    mdx = np.where(yrs==thisyr)
    print(str(thisyr) + ': '+loco+' ' + str(np.shape(mdx)[1]))

    if np.shape(mdx)[1] < 365:
        dt = ot[mdx]
        t1 = dt[0]
        t2 = dt[-1]

        start_date = str(thisyr)+'-01-01'
        end_date = str(thisyr)+'-12-31'
        date_array = pd.date_range(start=start_date, end=end_date, freq='D')
        dti = date_array + timedelta(hours=12)

        if dt[-1] == dti[-1]: 
            j = np.where(t1==dti)
            print(str(thisyr) + ' missing start of year')
            jj = int(j[0]) # why does this make an error for future in python ?? 
            
            missingtime = dti[:jj]
            dtj = np.concatenate((missingtime, dt), axis=None)
            NM = np.shape(missingtime)[0]
            nanarray = np.ones([NM,NZ])*np.nan

            u = np.concatenate((nanarray,ds.u.values[mdx,:].squeeze()), axis=0)
            v = np.concatenate((nanarray,ds.v.values[mdx,:].squeeze()), axis=0)
            w = np.concatenate((nanarray,ds.w.values[mdx,:].squeeze()), axis=0)
            e = np.concatenate((nanarray,ds.velprof.values[mdx,:].squeeze()), axis=0)
        else:
            j = np.where(t2==dti)
            print(str(thisyr) + ' missing end of year')
            jj = int(j[0])+1 # why does this make an error for future in python ?? 

            missingtime = dti[jj:]
            dtj = np.concatenate((dt,missingtime), axis=None)
            NM = np.shape(missingtime)[0]
            nanarray = np.ones([NM,NZ])*np.nan

            u = np.concatenate((ds.u.values[mdx,:].squeeze(),nanarray), axis=0)
            v = np.concatenate((ds.v.values[mdx,:].squeeze(),nanarray), axis=0)
            w = np.concatenate((ds.w.values[mdx,:].squeeze(),nanarray), axis=0)
            e = np.concatenate((ds.velprof.values[mdx,:].squeeze(),nanarray), axis=0)

        if np.all(dtj==dti):
            print('times fixed')
        else:
            print('time error')
            sys.exit()

        ds1 = xr.Dataset()
        ds1['ocean_time'] = (('ocean_time'), dtj, {'long_name':'daily average timestamps'})
        ds1['z'] = ds.z
        ds1['u'] = (('ocean_time','z'), u, {'units':'m.s-1', 'long_name': 'Daily avg OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity', 
                                      'source_file': fn_in, 'moored_location': ds.u.moored_location})
        ds1['v'] = (('ocean_time','z'), v, {'units':'m.s-1', 'long_name': 'Daily avg OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity', 
                                      'source_file': fn_in, 'moored_location': ds.v.moored_location})
        ds1['w'] = (('ocean_time','z'), w, {'units':'m.s-1', 'long_name': 'Daily avg Upward Sea Water Velocity','positive':'upward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity', 
                                      'source_file': fn_in, 'moored_location': ds.w.moored_location})
        ds1['velprof'] = (('ocean_time','z'), e, {'units':'m.s-1', 'long_name': 'Daily avg Error Seawater Velocity detail link broken','positive':'unclear',
                                      'metadata link':'BROKEN: https://mmisw.org/ont/cf/parameter/VELPROF-EVL', 
                                      'source_file': fn_in})
        
        fn_name = str(moor) + '_' + loco + '_ADCP_daily_'+str(thisyr)+'.01.01_'+str(thisyr)+'.12.31.nc'
        fn_o = posixpath.join(out_dir, fn_name)
        ds1.to_netcdf(fn_o, unlimited_dims='ocean_time')
        print('saved:' + str(thisyr))
        
    else:
        ds1 = xr.Dataset()
        ds1['ocean_time'] = (('ocean_time'), ot[mdx], {'long_name':'daily average timestamps'})
        ds1['z'] = ds.z
        ds1['u'] = ds.u[mdx] 
        ds1['v'] = ds.v[mdx] 
        ds1['w'] = ds.w[mdx] 
        ds1['velprof'] = ds.velprof[mdx] 

        ds1['u'].attrs['source_file'] = fn_in
        ds1['v'].attrs['source_file'] = fn_in
        ds1['w'].attrs['source_file'] = fn_in

        fn_name = str(moor) + '_' + loco + '_ADCP_daily_'+str(thisyr)+'.01.01_'+str(thisyr)+'.12.31.nc'
        fn_o = posixpath.join(out_dir, fn_name)
        ds1.to_netcdf(fn_o, unlimited_dims='ocean_time')
        print('saved:' + str(thisyr))


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
