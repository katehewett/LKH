'''
First step for CE09OSSM

This code is to process data from OOI for the WA surface mooring pressure data.

'''

import sys
import os
import xarray as xr
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#############################################################
# define a find duplicate function for QC of vel data below 
# we know there are duplicate values from looking at the data 
def find_duplicate_indices(list_):
    seen = {}
    duplicates = []
    for i, item in enumerate(list_):
        if item in seen:
            duplicates.append(i)
        else:
            seen[item] = True
    return duplicates
#############################################################

moor = 'CE09OSSM'
#loco = 'surfacebuoy'
loco = 'nsif'

if loco == 'nsif':
    ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/nsif/ooi-ce09ossm-rid26-04-velpta000_21b5_1859_99b7.nc', decode_times=True)
    out_dir = '/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/nsif'
elif loco == 'surfacebuoy':
    ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/surfacebuoy/ooi-ce09ossm-sbd11-04-velpta000_466c_6a3c_a582.nc', decode_times=True)
    out_dir = '/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/surfacebuoy'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)

zu = np.unique(ds.z.values)
ot = np.unique(ds.time.values)
NT = np.shape(ot)[0]
NZ = np.shape(zu)[0]

print('unique z, NZ: '+ str(NZ))
print('unique time, NT: '+ str(NT))
print('length z: ' + str(np.shape(ds.z.values)[0]))
print('length time: ' + str(np.shape(ds.time.values)[0]))

###############################################
# OOI download provides data in a long 1D array 
# want to convert to row/z and col/time
df = pd.DataFrame({'datetimes':ds.time.values})
df['date'] = df['datetimes'].dt.date

df['z'] = ds.z.values
if loco =='nsif':
    df['z'] = df['z']+-7 
if loco == 'surfacebuoy':
    df['z'] = df['z']+-1
df['u'] = ds.eastward_sea_water_velocity.values
df['v'] = ds.northward_sea_water_velocity.values
df['w'] = ds.upward_sea_water_velocity.values
#df['velprof'] = ds.velprof_evl.values

# check timestamps for duplicate entries
duplicates = df.duplicated(subset='datetimes')
if np.all(~duplicates):
    print('no duplicates in timestamp')

###############################################################################################################################
# remove big spikes 
ub = ds.eastward_sea_water_velocity.values
umean = np.nanmean(ub,axis=0)
ustd = np.nanstd(ub,axis=0)
uhigh = umean+7*ustd
ulow = umean-7*ustd
ub[(ub<ulow) | (ub>uhigh)] = np.nan

vb = ds.northward_sea_water_velocity.values
vmean = np.nanmean(vb,axis=0)
vstd = np.nanstd(vb,axis=0)
vhigh = vmean+7*vstd
vlow = vmean-7*vstd
vb[(vb<vlow) | (vb>vhigh)] = np.nan

wb = ds.upward_sea_water_velocity.values
wmean = np.nanmean(wb,axis=0)
wstd = np.nanstd(wb,axis=0)
whigh = wmean+7*wstd
wlow = wmean-7*wstd
wb[(wb<wlow) | (wb>whigh)] = np.nan

condition1 = (wb<wlow) | (wb>whigh)
condition2 = (vb<vlow) | (vb>vhigh)
condition3 = (ub<ulow) | (ub>uhigh)
conditions =  condition2 | condition3 

ub[conditions==True] = np.nan 
vb[conditions==True] = np.nan 
wb[conditions==True] = np.nan 
wb[condition1==True] = np.nan

plt.plot(ot,df['u'],'k.-')
plt.plot(ot,ub,'r.-')

#####################################################################
print('putting to ds...')

VELPTA = xr.Dataset()
VELPTA['ocean_time'] = (('ocean_time'), ot, {'long_name':'times from OOI, input as from 01-JAN-1970 00:00'})

z = df['z'].to_numpy 
VELPTA['z'] = (('ocean_time'), z, {'units':'m', 'long_name': 'OOI altitude ~depth below surface',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'})
VELPTA['u'] = (('ocean_time'), ub, {'units':'m.s-1', 'long_name': 'OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'})
VELPTA['v'] = (('ocean_time'), vb, {'units':'m.s-1', 'long_name': 'OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity'})
VELPTA['w'] = (('ocean_time'), wb, {'units':'m.s-1', 'long_name': 'Upward Sea Water Velocity','positive':'upward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity'})

if loco == 'nsif':
    VELPTA['u'].attrs['moored_location'] = 'nsif ~7m below surface'
    VELPTA['v'].attrs['moored_location'] = 'nsif ~7m below surface'
    VELPTA['w'].attrs['moored_location'] = 'nsif ~7m below surface'
    fn = out_dir + '/' + str(moor) + '_nsif_VELPTA.nc'
elif loco == 'surfacebuoy':
    VELPTA['u'].attrs['moored_location'] = 'surface buoy ~1m below surface'
    VELPTA['v'].attrs['moored_location'] = 'surface buoy ~1m below surface'
    VELPTA['w'].attrs['moored_location'] = 'surface buoy ~1m below surface'
    fn = out_dir + '/' + str(moor) + '_surfacebuoy_VELPTA.nc'

VELPTA.to_netcdf(fn, unlimited_dims='ocean_time')

print('saved!')