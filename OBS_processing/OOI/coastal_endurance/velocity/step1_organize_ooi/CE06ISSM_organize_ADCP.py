'''
First step for CE06ISSM
CE06ISSM_organize_ADCPs.py

This code is to process data from OOI for the WA surface mooring velocity data.

The ADCP data has already been binned by OOI/Axiom and is provides u,v,w (and error, see README)

CE06ISSM has 3495 w/c sets of duplicate values 

'/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd'
ooi-ce06issm-mfd35-01-vel3dd000_21b5_1859_99b7.nc  ooi-ce06issm-mfd35-04-adcptm000_dad4_820b_2d26.nc

this is ugly inflexible code, improve later :O
(after tried to rerun all processing files after 8 june 2025-- run time was super long. 
Changed some code to speed up...)
'''

import sys
import os
import xarray as xr
import numpy as np
import pandas as pd 
from datetime import datetime
from scipy import stats
import posixpath 
import warnings

from lo_tools import Lfun

Ldir = Lfun.Lstart()

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

error_threshold = 0.1

moor = 'CE06ISSM'
loco = 'mfd'

in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco 
fn_in = posixpath.join(in_dir, 'ooi-ce06issm-mfd35-04-adcptm000_dad4_820b_2d26.nc') 
ds = xr.open_dataset(fn_in, decode_times=True)

out_dir = Ldir['parent'] / 'LKH_data'/'OOI'/'CE'/'coastal_moorings'/moor/'velocity'/loco

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
df = pd.DataFrame({'datetimes':ds.time.values})
df['date'] = df['datetimes'].dt.date

df['z'] = ds.z.values
df['u'] = ds.eastward_sea_water_velocity.values
df['v'] = ds.northward_sea_water_velocity.values
df['w'] = ds.upward_sea_water_velocity.values
df['velprof'] = ds.velprof_evl.values

# set vars to nan if > error threshold 
df['Econdition'] = abs(df['velprof']) > error_threshold 
df.loc[df['Econdition'],'u'] = np.nan # take all the u rows where Econdition == True and set to nan
df.loc[df['Econdition'],'v'] = np.nan
df.loc[df['Econdition'],'w'] = np.nan # error is for u,v, but flagging w. not sure what's best here...
df.loc[df['Econdition'],'velprof'] = np.nan

#############################################################################################################
# group by timestamps 
print('grouping by timestamp...')
Zgroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='z')  
# check timestamps for duplicate entries
duplicates = Zgroup.duplicated(subset='datetimes')
if np.all(~duplicates):
    print('no duplicates in timestamp')
elif np.any(duplicates): 
    print('duplicates in time')
    sys.exit()

#Zgroup['Zmin'] = Zgroup.apply(lambda row: np.nanmin(row['z']), axis=1)
#Zgroup['Zmax'] = Zgroup.apply(lambda row: np.nanmax(row['z']), axis=1)

Ugroup = df.groupby('datetimes')['u'].apply(list).reset_index(name='u')
Vgroup = df.groupby('datetimes')['v'].apply(list).reset_index(name='v')
Wgroup = df.groupby('datetimes')['w'].apply(list).reset_index(name='w')
Egroup = df.groupby('datetimes')['velprof'].apply(list).reset_index(name='velprof')

#############################################################################################################
# We know there are a lot of instances where velocities thru the water column
# are repeated 2x for one timestamp. The next several lines of code will:
# 1 - remove duplicates, and 2 - grab indicies of duplicates
# This isn't super fast, TODO: find a way to speed up the search. But also unique to this mooring, so okay.
print('identifying duplicates in z group ...')

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

'''
print('cleaning up ...')
# housekeeping
del Zgroup, Ugroup, Vgroup, Wgroup, Egroup
columns_to_drop = ['z', 'dup_ind', 'has_duplicates']
Zgroup_copy = Zgroup_copy.drop(columns_to_drop,axis=1)

columns_to_drop = ['u', 'dup_ind', 'has_duplicates']
Ugroup_copy = Ugroup_copy.drop(columns_to_drop,axis=1)

columns_to_drop = ['v', 'dup_ind', 'has_duplicates']
Vgroup_copy = Vgroup_copy.drop(columns_to_drop,axis=1)

columns_to_drop = ['w', 'dup_ind', 'has_duplicates']
Wgroup_copy = Wgroup_copy.drop(columns_to_drop,axis=1)

columns_to_drop = ['velprof', 'dup_ind', 'has_duplicates']
Egroup_copy = Egroup_copy.drop(columns_to_drop,axis=1)

# not clipping depths like mfd, will do with binning
# we're going to flatten these 
print('flattening dataframe ...')
Zgroup_copy = Zgroup_copy.reset_index()
Ugroup_copy = Ugroup_copy.reset_index()
Vgroup_copy = Vgroup_copy.reset_index()
Wgroup_copy = Wgroup_copy.reset_index()
Egroup_copy = Egroup_copy.reset_index()
'''

Z = pd.DataFrame({'datetimes':Zgroup_copy['datetimes'],'Z':Zgroup_copy['new_value']})
U = pd.DataFrame({'datetimes':Ugroup_copy['datetimes'],'U':Ugroup_copy['new_value']})
V = pd.DataFrame({'datetimes':Vgroup_copy['datetimes'],'V':Vgroup_copy['new_value']})
W = pd.DataFrame({'datetimes':Wgroup_copy['datetimes'],'W':Wgroup_copy['new_value']})
E = pd.DataFrame({'datetimes':Egroup_copy['datetimes'],'E':Egroup_copy['new_value']})

'''
Zflatdata = pd.DataFrame([(index, value) for (index, values)
                         in Z['Z'].items() for value in values],
                             columns = ['index','Z']).set_index('index')
Z = Z.drop('Z', axis=1).join(Zflatdata)

Uflatdata = pd.DataFrame([(index, value) for (index, values)
                         in U['U'].items() for value in values],
                             columns = ['index','U']).set_index('index')
U = U.drop('U', axis=1).join(Uflatdata)

Vflatdata = pd.DataFrame([(index, value) for (index, values)
                         in V['V'].items() for value in values],
                             columns = ['index','V']).set_index('index')
V = V.drop('V', axis=1).join(Vflatdata)

Wflatdata = pd.DataFrame([(index, value) for (index, values)
                         in W['W'].items() for value in values],
                             columns = ['index','W']).set_index('index')
W = W.drop('W', axis=1).join(Wflatdata)

Eflatdata = pd.DataFrame([(index, value) for (index, values)
                         in E['E'].items() for value in values],
                             columns = ['index','E']).set_index('index')
E = E.drop('E', axis=1).join(Eflatdata)
'''


'''
#len(W)==len(U)==len(Z)==len(E)==len(V)    
plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

ax = plt.subplot2grid((4,4), (0,0), colspan=4,rowspan=1)
plt.plot(U['datetimes'],U['U'],color = 'grey',marker='.',linestyle='-',linewidth=2,alpha=0.8,markeredgecolor='none',label ='dups removed')
ax.set_title(moor + ' ADCP ' + loco)
ax.set_ylabel('u m/s')

ax1 = plt.subplot2grid((4,4), (1,0), colspan=4,rowspan=1)
plt.plot(V['datetimes'],V['V'],color = 'grey',marker='.',linestyle='-',linewidth=2,alpha=0.8,markeredgecolor='none',label ='dups removed')
ax1.set_ylabel('v m/s')

ax2 = plt.subplot2grid((4,4), (2,0), colspan=4,rowspan=1)
plt.plot(V['datetimes'],W['W'],color = 'grey',marker='.',linestyle='-',linewidth=2,alpha=0.8,markeredgecolor='none',label ='dups removed')
ax2.set_ylabel('w m/s')

ax3 = plt.subplot2grid((4,4), (3,0), colspan=4,rowspan=1)
plt.plot(E['datetimes'],E['E'],color = 'grey',marker='.',linestyle='-',linewidth=2,alpha=0.8,markeredgecolor='none',label ='dups removed')
ax3.set_ylabel('E m/s')
'''
###############################################################################################################################
#grid data 
print('gridding data ...')
Zcenter = np.arange(-29,0,1)
binedges = Zcenter-0.5
binedges = np.append(binedges,binedges[-1]+1)

NZc = np.shape(Zcenter)[0]
NTc = np.shape(Z['datetimes'].unique())[0]

dti = Z['datetimes'].unique()
print('NTc: '+str(NTc))

zb = np.ones([NTc,NZc])*np.nan
ub = np.ones([NTc,NZc])*np.nan
vb = np.ones([NTc,NZc])*np.nan
wb = np.ones([NTc,NZc])*np.nan
eb = np.ones([NTc,NZc])*np.nan

'''
Zgroupi = Z.groupby('datetimes')['Z'].apply(list).reset_index(name='z')
Ugroupi = U.groupby('datetimes')['Unew'].apply(list).reset_index(name='u')
Vgroupi = V.groupby('datetimes')['Vnew'].apply(list).reset_index(name='v')
Wgroupi = W.groupby('datetimes')['Wnew'].apply(list).reset_index(name='w')
Egroupi = E.groupby('datetimes')['Enew'].apply(list).reset_index(name='velprof')
'''

for idx in range(NTc):
    ui = U['U'][idx] 
    vi = V['V'][idx] 
    wi = W['W'][idx] 
    ei = E['E'][idx] 
    zi = Z['Z'][idx] 

    #ui = Ugroupi['u'].iloc[idx]
    #vi = Vgroupi['v'].iloc[idx]
    #wi = Wgroupi['w'].iloc[idx]
    #ei = Egroupi['e'].iloc[idx]
    #zi = Zgroupi['z'].iloc[idx]
    #statistic, bin_edges, binnumber = stats.binned_statistic(x, values, statistic='mean', bins=bins)
    #x the data to be binned; values the values on which the statistic will be computed 
    # values must have the same shape as x or be a list of arrays with the same shape as x)
    # x = [1, 1, 2, 5, 7] # values = [[1.0, 1.0, 2.0, 1.5, 3.0], [2.0, 2.0, 4.0, 3.0, 6.0]]
    # bins = [1, 4, 7] # statistic >> array([[1.33333333, 2.25],[2.66666667, 4.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ustat, bin_edges, binnumber = stats.binned_statistic(zi, ui, statistic=np.nanmean, bins=binedges)
        vstat, bin_edges, binnumber = stats.binned_statistic(zi, vi, statistic=np.nanmean, bins=binedges)
        wstat, bin_edges, binnumber = stats.binned_statistic(zi, wi, statistic=np.nanmean, bins=binedges)
        estat, bin_edges, binnumber = stats.binned_statistic(zi, ei, statistic=np.nanmean, bins=binedges)

    zb[idx,:] = Zcenter # just a checker
    ub[idx,:] = ustat
    vb[idx,:] = vstat
    wb[idx,:] = wstat
    eb[idx,:] = estat

###############################################################################################################################
# remove big spikes 
print('removing outliers...')
umean = np.nanmean(ub,axis=0)
ustd = np.nanstd(ub,axis=0)
uhigh = umean+7*ustd
ulow = umean-7*ustd
ub[(ub<ulow) | (ub>uhigh)] = np.nan

vmean = np.nanmean(vb,axis=0)
vstd = np.nanstd(vb,axis=0)
vhigh = vmean+7*vstd
vlow = vmean-7*vstd
vb[(vb<vlow) | (vb>vhigh)] = np.nan

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

###############################################################################################################################
# put to ds and save 
print('putting to ds...')
ADCP = xr.Dataset()
if np.all(Z['datetimes'].values==U['datetimes'].values):
    ADCP['ocean_time'] = (('ocean_time'), Z['datetimes'].values, {'long_name':'datetimes from OOI (time origin: 01-JAN-1970 00:00:00)'})
else: 
    print('exited script; werid time errors')
    sys.exit()

ADCP['z'] = (('ocean_time','z'), zb, {'units':'m', 'long_name':'re-binned centers, altitudes from OOI', 'positive':'up'})

ADCP['u'] = (('ocean_time','z'), ub, {'units':'m.s-1', 'long_name': 'OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'})
ADCP['v'] = (('ocean_time','z'), vb, {'units':'m.s-1', 'long_name': 'OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity'})
ADCP['w'] = (('ocean_time','z'), wb, {'units':'m.s-1', 'long_name': 'Upward Sea Water Velocity','positive':'upward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity'})
ADCP['velprof'] = (('ocean_time','z'), eb, {'units':'m.s-1', 'long_name': 'Error Seawater Velocity detail link broken','positive':'unclear',
                                      'metadata link':'BROKEN: https://mmisw.org/ont/cf/parameter/VELPROF-EVL'})

ADCP['u'].attrs['moored_location'] = 'mfd, depth ~29m'
ADCP['v'].attrs['moored_location'] = 'mfd, depth ~29m'
ADCP['w'].attrs['moored_location'] = 'mfd, depth ~29m'
fn_o = posixpath.join(out_dir , str(moor) + '_mfd_ADCP.nc')

ADCP.to_netcdf(fn_o, unlimited_dims='ocean_time')

print('saved!')