'''
First step for CE09OSSM
CE09OSSM_organize_ADCPs.py

This code is to process data from OOI for the WA surface mooring velocity data.

The ADCP data has already been binned by OOI/Axiom and is provides u,v,w (and error, see README)
Upward facing on MFD; downward NSIF

'/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/mfd'
ooi-ce09ossm-mfd35-04-adcpsj000_ff2d_359a_11eb.nc

'/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/nsif'
ooi-ce09ossm-rid26-01-adcptc000_dad4_820b_2d26.nc

(1) there are duplicates like there were in CE06ISSM, + there are wacky depths in the upward adcp

this is ugly inflexible code, improve later :O
'''

import sys
import os
import xarray as xr
import numpy as np
import pandas as pd 
from datetime import datetime
from scipy import stats

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
loco = 'mfd'
#loco = 'nsif'

if loco == 'nsif':
    ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/nsif/ooi-ce09ossm-rid26-01-adcptc000_dad4_820b_2d26.nc', decode_times=True)
    out_dir = '/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/nsif'
elif loco == 'mfd':
    ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/mfd/ooi-ce09ossm-mfd35-04-adcpsj000_ff2d_359a_11eb.nc', decode_times=True)
    out_dir = '/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/mfd'

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

if loco == 'mfd':
    a_datetime = datetime(2015, 5, 2, 16, 00, 0) # clipping profiles before this time (suspect)
    idx_to_drop = df[(df['datetimes'] < a_datetime)].index 
    df = df.drop(idx_to_drop, inplace=False)

'''
if loco=='mfd': # there are data that are 500m or -8000m in this dataset. Removing those data here:
    idx_to_drop = df[(df['z'] > -50) | (df['z'] < -540)].index 
    df = df.drop(idx_to_drop, inplace=False)
    zu = np.unique(df['z'])
    NZ = np.shape(zu)[0]
    print('updated: unique z, NZ: '+ str(NZ))
'''
print('grouping by timestamp...')
#############################################################################################################
# group by timestamps 
Zgroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='z')  
# check timestamps for duplicate entries
duplicates = Zgroup.duplicated(subset='datetimes')
if np.all(~duplicates):
    print('no duplicates in timestamp')
elif np.any(duplicates): 
    print('duplicates in time')
    sys.exit()

#Drop if the ...
#min z is deeper than 540m  
#min z doesn't reach deeper than 495m 
#max z is shallower than -100m drop

Zgroup['Zmin'] = Zgroup.apply(lambda row: np.nanmin(row['z']), axis=1)
Zgroup['Zmax'] = Zgroup.apply(lambda row: np.nanmax(row['z']), axis=1)

idx2_to_drop = Zgroup[(Zgroup['Zmin'] < -540) | (Zgroup['Zmin'] > -495) | (Zgroup['Zmax'] < -100)].index 
Zgroup = Zgroup.drop(idx2_to_drop, inplace=False)

Ugroup = df.groupby('datetimes')['u'].apply(list).reset_index(name='u')
Vgroup = df.groupby('datetimes')['v'].apply(list).reset_index(name='v')
Wgroup = df.groupby('datetimes')['w'].apply(list).reset_index(name='w')
Egroup = df.groupby('datetimes')['velprof'].apply(list).reset_index(name='velprof')

Ugroup = Ugroup.drop(idx2_to_drop, inplace=False)
Vgroup = Vgroup.drop(idx2_to_drop, inplace=False)
Wgroup = Wgroup.drop(idx2_to_drop, inplace=False)
Egroup = Egroup.drop(idx2_to_drop, inplace=False)

'''
if loco=='mfd': 
    ot_new = np.unique(Zgroup['datetimes'])
    NT = np.shape(ot_new)[0]
    print('updated: unique time, NT: '+ str(NT))
'''

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

print('cleaning up ...')
# housekeeping
del Zgroup, Ugroup, Vgroup, Wgroup, Egroup
columns_to_drop = ['z', 'Zmin', 'Zmax', 'dup_ind', 'has_duplicates']
Zgroup_copy = Zgroup_copy.set_index('datetimes')
Zgroup_copy = Zgroup_copy.drop(columns_to_drop,axis=1)

columns_to_drop = ['u', 'dup_ind', 'has_duplicates']
Ugroup_copy = Ugroup_copy.set_index('datetimes')
Ugroup_copy = Ugroup_copy.drop(columns_to_drop,axis=1)

columns_to_drop = ['v', 'dup_ind', 'has_duplicates']
Vgroup_copy = Vgroup_copy.set_index('datetimes')
Vgroup_copy = Vgroup_copy.drop(columns_to_drop,axis=1)

columns_to_drop = ['w', 'dup_ind', 'has_duplicates']
Wgroup_copy = Wgroup_copy.set_index('datetimes')
Wgroup_copy = Wgroup_copy.drop(columns_to_drop,axis=1)

columns_to_drop = ['velprof', 'dup_ind', 'has_duplicates']
Egroup_copy = Egroup_copy.set_index('datetimes')
Egroup_copy = Egroup_copy.drop(columns_to_drop,axis=1)

print('surface clipping ...')
if loco=='mfd': # clipping shallower than -30m , just to make it smaller (we clip in the binedges step to -50m)
    Zgroup_copy['idx_to_drop'] = Zgroup_copy.apply(lambda row: np.where(row['new_value']>-30), axis=1) 
    Zgroup_copy['needs_clipp'] = Zgroup_copy['idx_to_drop'].apply(lambda x: len(x) > 0)
    Zgroup_copy['clipped_value'] = Zgroup_copy.apply(lambda row: np.delete(np.array(row['new_value']),row['idx_to_drop']) if row['needs_clipp'] else np.array(row['new_value']),axis=1)

    Ugroup_copy['needs_clipp']=Zgroup_copy['needs_clipp'].copy()
    Ugroup_copy['idx_to_drop'] = Zgroup_copy['idx_to_drop'].copy()
    Ugroup_copy['clipped_value'] = Ugroup_copy.apply(lambda row: np.delete(np.array(row['new_value']),row['idx_to_drop']) if row['needs_clipp'] else np.array(row['new_value']),axis=1)
    
    Vgroup_copy['needs_clipp']=Zgroup_copy['needs_clipp'].copy()
    Vgroup_copy['idx_to_drop'] = Zgroup_copy['idx_to_drop'].copy()
    Vgroup_copy['clipped_value'] = Vgroup_copy.apply(lambda row: np.delete(np.array(row['new_value']),row['idx_to_drop']) if row['needs_clipp'] else np.array(row['new_value']),axis=1)
    
    Wgroup_copy['needs_clipp']=Zgroup_copy['needs_clipp'].copy()
    Wgroup_copy['idx_to_drop'] = Zgroup_copy['idx_to_drop'].copy()
    Wgroup_copy['clipped_value'] = Wgroup_copy.apply(lambda row: np.delete(np.array(row['new_value']),row['idx_to_drop']) if row['needs_clipp'] else np.array(row['new_value']),axis=1)
    
    Egroup_copy['needs_clipp']=Zgroup_copy['needs_clipp'].copy()
    Egroup_copy['idx_to_drop'] = Zgroup_copy['idx_to_drop'].copy()
    Egroup_copy['clipped_value'] = Egroup_copy.apply(lambda row: np.delete(np.array(row['new_value']),row['idx_to_drop']) if row['needs_clipp'] else np.array(row['new_value']),axis=1)

#Zgroup_copy['spacing'] = Zgroup.apply(lambda row: np.diff(row['clipped_value']), axis=1)
#len(Zgroup_copy) == len(Ugroup_copy) == len(Vgroup_copy) == len(Wgroup_copy)

# we're going to flatten these 
print('flattening dataframe ...')
Zgroup_copy = Zgroup_copy.reset_index()
Ugroup_copy = Ugroup_copy.reset_index()
Vgroup_copy = Vgroup_copy.reset_index()
Wgroup_copy = Wgroup_copy.reset_index()
Egroup_copy = Egroup_copy.reset_index()

Z = pd.DataFrame({'datetimes':Zgroup_copy['datetimes'],'Z':Zgroup_copy['clipped_value']})
U = pd.DataFrame({'datetimes':Ugroup_copy['datetimes'],'U':Ugroup_copy['clipped_value']})
V = pd.DataFrame({'datetimes':Vgroup_copy['datetimes'],'V':Vgroup_copy['clipped_value']})
W = pd.DataFrame({'datetimes':Wgroup_copy['datetimes'],'W':Wgroup_copy['clipped_value']})
E = pd.DataFrame({'datetimes':Egroup_copy['datetimes'],'E':Egroup_copy['clipped_value']})

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

condition1 = abs(E['E'])>0.1
condition2 = abs(U['U'])>0.9
condition3 = abs(V['V'])>1.5
condition4 = W['W']<-0.4
conditions =  condition1 | condition2 | condition3 | condition4 

U['QC'] = conditions
result = U[U['QC']==True]
U['Unew'] = np.where(U['QC'] ==True, np.nan, U['U'])

V['QC'] = conditions
result = V[V['QC']==True]
V['Vnew'] = np.where(V['QC'] ==True, np.nan, V['V'])

W['QC'] = conditions
result = W[W['QC']==True]
W['Wnew'] = np.where(W['QC'] ==True, np.nan, W['W'])

E['QC'] = conditions
result = E[E['QC']==True]
E['Enew'] = np.where(E['QC'] ==True, np.nan, E['E'])

'''
ax.plot(U['datetimes'],U['Unew'],color = 'orange',marker='.',linestyle='none',alpha=0.8,markeredgecolor='none',label ='QC')
ax1.plot(V['datetimes'],V['Vnew'],color = 'orange',marker='.',linestyle='none',alpha=0.8,markeredgecolor='none',label ='QC')
ax2.plot(W['datetimes'],W['Wnew'],color = 'orange',marker='.',linestyle='none',alpha=0.8,markeredgecolor='none',label ='QC')
ax3.plot(E['datetimes'],E['Enew'],color = 'orange',marker='.',linestyle='none',alpha=0.8,markeredgecolor='none',label ='QC')

ax.legend(loc="best",frameon=False)
ax1.legend(loc="best",frameon=False)
ax2.legend(loc="best",frameon=False)
ax3.legend(loc="best",frameon=False)
'''

#grid interp data 
print('gridding data ...')
Zcenter = np.arange(-530,-49,15)  # originally went to 
binedges = Zcenter-7.5
binedges = np.append(binedges,binedges[-1]+15)

NZc = np.shape(Zcenter)[0]
NTc = np.shape(Z['datetimes'].unique())[0]

zb = np.ones([NTc,NZc])*np.nan
ub = np.ones([NTc,NZc])*np.nan
vb = np.ones([NTc,NZc])*np.nan
wb = np.ones([NTc,NZc])*np.nan
eb = np.ones([NTc,NZc])*np.nan

Zgroupi = Z.groupby('datetimes')['Z'].apply(list).reset_index(name='z')
Ugroupi = U.groupby('datetimes')['Unew'].apply(list).reset_index(name='u')
Vgroupi = V.groupby('datetimes')['Vnew'].apply(list).reset_index(name='v')
Wgroupi = W.groupby('datetimes')['Wnew'].apply(list).reset_index(name='w')
Egroupi = E.groupby('datetimes')['Enew'].apply(list).reset_index(name='velprof')

for idx in range(NTc):
    ui = Ugroupi['u'].iloc[idx]
    vi = Vgroupi['v'].iloc[idx]
    wi = Wgroupi['w'].iloc[idx]
    ei = Egroupi['velprof'].iloc[idx]
    zi = Zgroupi['z'].iloc[idx]
    #statistic, bin_edges, binnumber = stats.binned_statistic(x, values, statistic='mean', bins=bins)
    #x the data to be binned; values the values on which the statistic will be computed 
    # values must have the same shape as x or be a list of arrays with the same shape as x)
    # x = [1, 1, 2, 5, 7] # values = [[1.0, 1.0, 2.0, 1.5, 3.0], [2.0, 2.0, 4.0, 3.0, 6.0]]
    # bins = [1, 4, 7] # statistic >> array([[1.33333333, 2.25],[2.66666667, 4.5]])
    ustat, bin_edges, binnumber = stats.binned_statistic(zi, ui, statistic='mean', bins=binedges)
    vstat, bin_edges, binnumber = stats.binned_statistic(zi, vi, statistic='mean', bins=binedges)
    wstat, bin_edges, binnumber = stats.binned_statistic(zi, wi, statistic='mean', bins=binedges)
    estat, bin_edges, binnumber = stats.binned_statistic(zi, ei, statistic='mean', bins=binedges)

    zb[idx,:] = Zcenter # just a checker
    ub[idx,:] = ustat
    vb[idx,:] = vstat
    wb[idx,:] = wstat
    eb[idx,:] = estat

print('putting to ds...')

ADCP = xr.Dataset()
if np.all(Zgroupi['datetimes'].values==Ugroupi['datetimes'].values):
    ADCP['ocean_time'] = (('ocean_time'), Zgroupi['datetimes'].values, {'long_name':'datetimes from OOI, 01-JAN-1970 00:00:00'})
else: 
    print('exited script; werid time errors')
    sys.exit()

ADCP['z'] = (('z'), zb, {'units':'m', 'long_name':'altitude from OOI', 'positive':'up'})

ADCP['u'] = (('ocean_time','z'), ub, {'units':'m.s-1', 'long_name': 'OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'})
ADCP['v'] = (('ocean_time','z'), vb, {'units':'m.s-1', 'long_name': 'OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity'})
ADCP['w'] = (('ocean_time','z'), wb, {'units':'m.s-1', 'long_name': 'Upward Sea Water Velocity','positive':'upward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity'})
ADCP['velprof'] = (('ocean_time','z'), eb, {'units':'m.s-1', 'long_name': 'Error Seawater Velocity detail link broken','positive':'unclear',
                                      'metadata link':'BROKEN: https://mmisw.org/ont/cf/parameter/VELPROF-EVL'})

if loco == 'nsif':
    ADCP['u'].attrs['moored_location'] = 'nsif ~7m below surface'
    ADCP['v'].attrs['moored_location'] = 'nsif ~7m below surface'
    ADCP['w'].attrs['moored_location'] = 'nsif ~7m below surface'
    fn_o = out_dir + '/' + str(moor) + '_nsif_ADCP.nc'
elif loco == 'mfd':
    ADCP['u'].attrs['moored_location'] = 'mfd, depth ~540m'
    ADCP['v'].attrs['moored_location'] = 'mfd, depth ~540m'
    ADCP['w'].attrs['moored_location'] = 'mfd, depth ~540m'
    fn_o = out_dir + '/' + str(moor) + '_mfd_ADCP.nc'

ADCP.to_netcdf(fn_o, unlimited_dims='ocean_time')

print('saved!')

