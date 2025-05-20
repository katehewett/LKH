'''
First step for CE06ISSM
CE06ISSM_organize_VEL3D.py

This code is to process data from OOI for the WA surface mooring velocity data.

The ADCP data has already been binned by OOI/Axiom and is provides u,v,w (and error, see README)

'/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd'
ooi-ce06issm-mfd35-01-vel3dd000_21b5_1859_99b7.nc  
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

moor = 'CE06ISSM'

ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd/ooi-ce06issm-mfd35-01-vel3dd000_21b5_1859_99b7.nc', decode_times=True)

fn_o = '/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd'
if os.path.exists(fn_o)==False:
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
df['u'] = ds.eastward_sea_water_velocity.values
df['v'] = ds.northward_sea_water_velocity.values
df['w'] = ds.upward_sea_water_velocity.values
#df['velprof'] = ds.velprof_evl.values

# check timestamps for duplicate entries
duplicates = df.duplicated(subset='datetimes')
if np.all(~duplicates):
    print('no duplicates in timestamp')

#####################################################################
print('putting to ds...')

VEL3D = xr.Dataset()
VEL3D['ocean_time'] = (('ocean_time'), ot, {'long_name':'times from OOI, input as from 01-JAN-1970 00:00'})

VEL3D['u'] = (('ocean_time'), df['u'], {'units':'m.s-1', 'long_name': 'OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'})
VEL3D['v'] = (('ocean_time'), df['v'], {'units':'m.s-1', 'long_name': 'OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity'})
VEL3D['w'] = (('ocean_time'), df['w'], {'units':'m.s-1', 'long_name': 'Upward Sea Water Velocity','positive':'upward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity'})

VEL3D['moored_location'] = 'mfd ~ 29m'

fn = fn_o + '/' + str(moor) + '_mfd_VEL3D.nc'
VEL3D.to_netcdf(fn, unlimited_dims='ocean_time')

print('saved!')

sys.exit()

'''
plt.plot(zb[:,19458],ub[:,19458])
plt.plot(Zgroup_copy['z'][19458],Ugroup_copy['u'][19458])

s=Zgroup_copy['new_value']
s2 = np.concatenate(s)
np.nansum(s2)
np.nansum(zb)

s=Ugroup_copy['new_value']
s2 = np.concatenate(s)
np.nansum(s2)
np.nansum(ub)

s=Vgroup_copy['new_value']
s2 = np.concatenate(s)
np.nansum(s2)
np.nansum(vb)

s=Wgroup_copy['new_value']
s2 = np.concatenate(s)
np.nansum(s2)
np.nansum(wb)
'''



'''
Pmfn = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/pressure/mfd/ooi-ce06issm-mfd37-03-ctdbpc000_5d2b_5c05_8b78.nc', decode_times=True)
#Pnsif = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/pressure/nsif/ooi-ce09ossm-rid27-03-ctdbpc000_9def_2170_af4e.nc', decode_times=True)
Pseafloor = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/pressure/mfd/ooi-ce06issm-mfd35-02-presfa000_2d0a_0cb6_f73c.nc', decode_times=True)

df_p = pd.DataFrame({'datetimes':Pmf.time.values})


ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/mfd/ooi-ce09ossm-mfd35-04-adcpsj000_ff2d_359a_11eb.nc', decode_times=True)                   
Pmfn = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/pressure/mfd/ooi-ce09ossm-mfd37-03-ctdbpe000_9def_2170_af4e.nc', decode_times=True)
Pnsif = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/pressure/nsif/ooi-ce09ossm-rid27-03-ctdbpc000_9def_2170_af4e.nc', decode_times=True)
Pseafloor = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/pressure/mfd/ooi-ce09ossm-mfd35-02-presfc000_e0c5_bc24_0ec6.nc', decode_times=True)


#format plotting space 
plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 6.5
width_of_image = 10
fig1 = plt.figure(figsize=(width_of_image,height_of_image))
fig1.set_size_inches(width_of_image,height_of_image, forward=False)

ax = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
plt.plot(Pseafloor.time.values[Pseafloor.sea_water_pressure_at_sea_floor_tide_qc_agg==1],Pseafloor.sea_water_pressure_at_sea_floor_tide.values[Pseafloor.sea_water_pressure_at_sea_floor_tide_qc_agg==1],'k.',label = 'sea floor')
plt.plot(Pmfn.time.values[Pmfn.sea_water_pressure_qc_agg==1],Pmfn.sea_water_pressure.values[Pmfn.sea_water_pressure_qc_agg==1],'b.',label = 'mfn')
plt.legend(loc="best")
ax.set_ylabel('pressure decibars')
ax.set_title(moor + '  only QC = 1')
plt.gcf().tight_layout()

dsw = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/wave_sw/surfacebuoy/ooi-ce09ossm-sbd12-05-wavssa000_5364_5a59_7501.nc', decode_times=True)
ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
plt.plot(dsw.time.values,dsw.sea_surface_wave_maximum_height.values,'g.',label = 'all')
plt.plot(dsw.time.values[dsw.sea_surface_wave_maximum_height_qc_agg==1],dsw.sea_surface_wave_maximum_height.values[dsw.sea_surface_wave_maximum_height_qc_agg==1],'m.',label = 'qc=1')
plt.legend()
ys.exit()                   
'''