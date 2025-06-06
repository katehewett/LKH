'''
First step for CE06ISSM
CE06ISSM_organize_ADCPs.py

This code is to process data from OOI for the WA surface mooring velocity data.

The ADCP data has already been binned by OOI/Axiom and is provides u,v,w (and error, see README)

CE06ISSM has 3495 w/c sets of duplicate values 

'/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd'
ooi-ce06issm-mfd35-01-vel3dd000_21b5_1859_99b7.nc  ooi-ce06issm-mfd35-04-adcptm000_dad4_820b_2d26.nc
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

ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd/ooi-ce06issm-mfd35-04-adcptm000_dad4_820b_2d26.nc', decode_times=True)

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
df['velprof'] = ds.velprof_evl.values

print('grouping by day...')

#############################################################################################################
# group by timestamps 
Zgroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='z')  
Ugroup = df.groupby('datetimes')['u'].apply(list).reset_index(name='u')
Vgroup = df.groupby('datetimes')['v'].apply(list).reset_index(name='v')
Wgroup = df.groupby('datetimes')['w'].apply(list).reset_index(name='w')
Egroup = df.groupby('datetimes')['velprof'].apply(list).reset_index(name='velprof')

# check timestamps for duplicate entries
duplicates = Zgroup.duplicated(subset='datetimes')
if np.all(~duplicates):
    print('no duplicates in timestamp')

#############################################################################################################
# We know there are a lot of instances where velocities thru the water column
# are repeated for one timestamp. The next several lines of code will:
# 1 - remove duplicates, and 2 - grab indicies of duplicates
# This isn't super fast, TODO: find a way to speed up the search. But also unique to this mooring, so okay.
print('identifying duplicates in z ...')

Zgroup_copy = Zgroup.copy()
Zgroup_copy['dup_ind']=Zgroup_copy['z'].apply(find_duplicate_indices)
Zgroup_copy['has_duplicates'] = Zgroup_copy['dup_ind'].apply(lambda x: len(x) > 0)

print('removing duplicates in z ...')
Zgroup_copy['new_value'] = Zgroup_copy.apply(lambda row: np.array(row['z'])[row['dup_ind']] if row['has_duplicates'] else np.array(row['z']), axis=1)

print('removing duplicates in u ...')
Ugroup_copy = Ugroup.copy()
Ugroup_copy['dup_ind']=Zgroup_copy['dup_ind'].copy()
Ugroup_copy['has_duplicates'] = Zgroup_copy['has_duplicates'].copy()
Ugroup_copy['new_value'] = Ugroup_copy.apply(lambda row: np.array(row['u'])[row['dup_ind']] if row['has_duplicates'] else np.array(row['u']), axis=1)

print('removing duplicates in v ...')
Vgroup_copy = Vgroup.copy()
Vgroup_copy['dup_ind']=Zgroup_copy['dup_ind'].copy()
Vgroup_copy['has_duplicates'] = Zgroup_copy['has_duplicates'].copy()
Vgroup_copy['new_value'] = Vgroup_copy.apply(lambda row: np.array(row['v'])[row['dup_ind']] if row['has_duplicates'] else np.array(row['v']), axis=1)

print('removing duplicates in w ...')
Wgroup_copy = Wgroup.copy()
Wgroup_copy['dup_ind']=Zgroup_copy['dup_ind'].copy()
Wgroup_copy['has_duplicates'] = Zgroup_copy['has_duplicates'].copy()
Wgroup_copy['new_value'] = Wgroup_copy.apply(lambda row: np.array(row['w'])[row['dup_ind']] if row['has_duplicates'] else np.array(row['w']), axis=1)

print('removing duplicates in velprof ...')
Egroup_copy = Egroup.copy()
Egroup_copy['dup_ind']=Zgroup_copy['dup_ind'].copy()
Egroup_copy['has_duplicates'] = Zgroup_copy['has_duplicates'].copy()
Egroup_copy['new_value'] = Egroup_copy.apply(lambda row: np.array(row['velprof'])[row['dup_ind']] if row['has_duplicates'] else np.array(row['velprof']), axis=1)

print('gridding data ...')
#####################################################################
# Organize arrays s/t they are ready to be placed in a gridded format 
Zgroup_copy['zu_bool'] = Zgroup_copy.apply(lambda row: np.isin(zu,row['new_value']), axis=1)
Zgroup_copy['zu_idx'] = Zgroup_copy.apply(lambda row: np.where(row['zu_bool']), axis=1)

# id where the z's are in the full bin range, zu
zbool = np.ones([NZ,NT])*np.nan
zb = np.ones([NZ,NT])*np.nan
ub = np.ones([NZ,NT])*np.nan
vb = np.ones([NZ,NT])*np.nan
wb = np.ones([NZ,NT])*np.nan
eb = np.ones([NZ,NT])*np.nan

for i in range(NT):
    zbool[:,i] = Zgroup_copy['zu_bool'][i]

    zb[Zgroup_copy['zu_idx'][i],i] = Zgroup_copy['new_value'][i]
    ub[Zgroup_copy['zu_idx'][i],i] = Ugroup_copy['new_value'][i]
    vb[Zgroup_copy['zu_idx'][i],i] = Vgroup_copy['new_value'][i]
    wb[Zgroup_copy['zu_idx'][i],i] = Wgroup_copy['new_value'][i]
    eb[Zgroup_copy['zu_idx'][i],i] = Egroup_copy['new_value'][i]

print('putting to ds...')

ADCP = xr.Dataset()
if np.all(Zgroup_copy['datetimes'].values==ot):
    ADCP['ocean_time'] = (('ocean_time'), ot, {'long_name':'times from OOI, input as from 01-JAN-1970 00:00'})
else: 
    print('exited script; werid time errors')
    sys.exit()

ADCP['z'] = (('z'), zu, {'units':'m', 'long_name':'altitude from OOI', 'positive':'up'})

ADCP['zbool'] = (('z','ocean_time'), zbool, {'units':'bool', 'long_name': '0 = no data; 1 = data'})
ADCP['u'] = (('z','ocean_time'), ub, {'units':'m.s-1', 'long_name': 'OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'})
ADCP['v'] = (('z','ocean_time'), vb, {'units':'m.s-1', 'long_name': 'OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity'})
ADCP['w'] = (('z','ocean_time'), wb, {'units':'m.s-1', 'long_name': 'Upward Sea Water Velocity','positive':'upward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity'})
ADCP['velprof'] = (('z','ocean_time'), eb, {'units':'m.s-1', 'long_name': 'Error Seawater Velocity detail link broken','positive':'unclear',
                                      'metadata link':'BROKEN: https://mmisw.org/ont/cf/parameter/VELPROF-EVL'})

fn = fn_o + '/' + str(moor) + '_mfd_ADCP.nc'
ADCP.to_netcdf(fn, unlimited_dims='ocean_time')

print('saved!')

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