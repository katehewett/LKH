'''
First step for CE07SHSM

This code is to process data from OOI for the WA surface mooring velocity data.

The ADCP data has already been binned by OOI/Axiom and is provides u,v,w (and error, see README)
Upward facing on MFD; downward NSIF

'/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE07SHSM/velocity/mfd'
ooi-ce09ossm-mfd35-04-adcpsj000_ff2d_359a_11eb.nc

'/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/nsif'
ooi-ce09ossm-rid26-01-adcptc000_dad4_820b_2d26.nc

this is ugly inflexible code, improve later :O
'''

import sys
import os
import xarray as xr
import numpy as np
import pandas as pd 
from datetime import datetime
from scipy import stats
import warnings

import matplotlib.pyplot as plt

###############################################################################################################################
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
###############################################################################################################################

error_threshold = 0.1

moor = 'CE07SHSM'
#loco = 'mfd'
loco = 'nsif'

if loco == 'nsif':
    ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE07SHSM/velocity/nsif/ooi-ce07shsm-rid26-01-adcpta000_dad4_820b_2d26.nc', decode_times=True)
    out_dir = '/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE07SHSM/velocity/nsif'
elif loco == 'mfd':
    ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE07SHSM/velocity/mfd/ooi-ce07shsm-mfd35-04-adcptc000_dad4_820b_2d26.nc', decode_times=True)
    out_dir = '/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE07SHSM/velocity/mfd'

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

###############################################################################################################################
# OOI download provides data in a long 1D array 
df = pd.DataFrame({'datetimes':ds.time.values})
df['date'] = df['datetimes'].dt.date

df['z'] = ds.z.values
df['u'] = ds.eastward_sea_water_velocity.values
df['v'] = ds.northward_sea_water_velocity.values
df['w'] = ds.upward_sea_water_velocity.values
df['velprof'] = ds.velprof_evl.values

######################################################################################################
'''
# check the data and a lot to see about the error term, there isn't a value calc'd for 2015 
if loco == 'mfd':
    Zcenter = np.arange(-530,-49,15)  
    binedges = Zcenter-7.5
    binedges = np.append(binedges,binedges[-1]+15)

if loco == 'nsif':
    Zcenter = np.arange(-95,-14,5) 
    binedges = Zcenter-2.5
    binedges = np.append(binedges,binedges[-1]+5)

#statistic, bin_edges, binnumber = stats.binned_statistic(x, values, statistic='mean', bins=bins)
#x the data to be binned; values the values on which the statistic will be computed 
# values must have the same shape as x or be a list of arrays with the same shape as x)
# x = [1, 1, 2, 5, 7] # values = [[1.0, 1.0, 2.0, 1.5, 3.0], [2.0, 2.0, 4.0, 3.0, 6.0]]
# bins = [1, 4, 7] # statistic >> array([[1.33333333, 2.25],[2.66666667, 4.5]])
Emean, bin_edges, binnumber = stats.binned_statistic(df['z'], df['velprof'], statistic=np.nanmean, bins=binedges)
Estd, bin_edges, binnumber = stats.binned_statistic(df['z'], df['velprof'], statistic=np.nanstd, bins=binedges)

plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

ax = plt.subplot2grid((4,4), (0,0), colspan=1,rowspan=4)
plt.plot(df['velprof'],df['z'],'b.')
plt.plot(Emean,Zcenter,color = 'grey',marker='none',linestyle='-',linewidth=2,alpha=0.8,label ='Emean')
plt.axvline(x=0.1, color='r', linestyle='--')
plt.axvline(x=-0.1, color='r', linestyle='--')
ax.set_ylabel('e m/s')

ax1 = plt.subplot2grid((4,4), (0,2), colspan=1,rowspan=4)
plt.plot(df['velprof'],df['z'],'b.')
plt.plot(Emean,Zcenter,color = 'grey',marker='none',linestyle='-',linewidth=2,alpha=0.8,label ='Emean')
plt.plot(Emean-Estd,Zcenter,color = 'grey',marker='none',linestyle='-',linewidth=2,alpha=0.8)
plt.plot(Emean+Estd,Zcenter,color = 'grey',marker='none',linestyle='-',linewidth=2,alpha=0.8)
plt.axvline(x=0.1, color='r', linestyle='--')
plt.axvline(x=-0.1, color='r', linestyle='--')
plt.axvline(x=0.05, color='c', linestyle='--')
plt.axvline(x=-0.05, color='c', linestyle='--')
ax1.set_ylim([-100,-5])
ax1.set_xlim([-0.5,0.5])
ax1.set_ylabel('e m/s')

sys.exit()
'''
######################################################################################################
# group by timestamps 
print('grouping by timestamp...')
Zgroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='z')  
# check timestamps for duplicate entries
duplicates = Zgroup.duplicated(subset='datetimes')
if np.all(~duplicates):
    print('no duplicate timestamps')
elif np.any(duplicates): 
    print('duplicates in time')
    sys.exit()

#mfd drop if the ...
#min z is deeper than 540m  
#min z doesn't reach deeper than 495m 
#max z is shallower than -100m drop
if loco == 'mfd':
    Zgroup['Zmin'] = Zgroup.apply(lambda row: np.nanmin(row['z']), axis=1)
    Zgroup['Zmax'] = Zgroup.apply(lambda row: np.nanmax(row['z']), axis=1)
    idx2_to_drop = Zgroup[(Zgroup['Zmin'] < -89) | (Zgroup['Zmax'] < -30)].index 
    Zgroup = Zgroup.drop(idx2_to_drop, inplace=False)

Ugroup = df.groupby('datetimes')['u'].apply(list).reset_index(name='u')
print('ugrouped')
Vgroup = df.groupby('datetimes')['v'].apply(list).reset_index(name='v')
print('vgrouped')
Wgroup = df.groupby('datetimes')['w'].apply(list).reset_index(name='w')
print('wgrouped')
Egroup = df.groupby('datetimes')['velprof'].apply(list).reset_index(name='velprof')
print('egrouped')

if loco == 'mfd':
    Ugroup = Ugroup.drop(idx2_to_drop, inplace=False)
    Vgroup = Vgroup.drop(idx2_to_drop, inplace=False)
    Wgroup = Wgroup.drop(idx2_to_drop, inplace=False)
    Egroup = Egroup.drop(idx2_to_drop, inplace=False)

###############################################################################################################################
# We know there are a lot of instances where velocities thru the water column
# are repeated 2x for one timestamp. The next several lines of code will:
# 1 - remove duplicates, and 2 - grab indicies of duplicates
# This isn't super fast, TODO: find a way to speed up the search. 
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


###############################################################################################################################
# housekeeping
print('cleaning up ...')
del Zgroup, Ugroup, Vgroup, Wgroup, Egroup
if loco == 'mfd':
    columns_to_drop = ['z', 'Zmin', 'Zmax', 'dup_ind', 'has_duplicates']
if loco == 'nsif':
    columns_to_drop = ['z', 'dup_ind', 'has_duplicates']
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

###############################################################################################################################
if loco == 'mfd':
    print('surface clipping ...')
    # clipping shallower than -30m , just to make it smaller (we clip in the binedges step w/ -50m center)
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

###############################################################################################################################
# we're going to flatten these 
print('flattening dataframe ...')
Zgroup_copy = Zgroup_copy.reset_index()
Ugroup_copy = Ugroup_copy.reset_index()
Vgroup_copy = Vgroup_copy.reset_index()
Wgroup_copy = Wgroup_copy.reset_index()
Egroup_copy = Egroup_copy.reset_index()

if loco == 'mfd':
    Z = pd.DataFrame({'datetimes':Zgroup_copy['datetimes'],'Z':Zgroup_copy['clipped_value']})
    U = pd.DataFrame({'datetimes':Ugroup_copy['datetimes'],'U':Ugroup_copy['clipped_value']})
    V = pd.DataFrame({'datetimes':Vgroup_copy['datetimes'],'V':Vgroup_copy['clipped_value']})
    W = pd.DataFrame({'datetimes':Wgroup_copy['datetimes'],'W':Wgroup_copy['clipped_value']})
    E = pd.DataFrame({'datetimes':Egroup_copy['datetimes'],'E':Egroup_copy['clipped_value']})
if loco == 'nsif':
    Z = pd.DataFrame({'datetimes':Zgroup_copy['datetimes'],'Z':Zgroup_copy['new_value']})
    U = pd.DataFrame({'datetimes':Ugroup_copy['datetimes'],'U':Ugroup_copy['new_value']})
    V = pd.DataFrame({'datetimes':Vgroup_copy['datetimes'],'V':Vgroup_copy['new_value']})
    W = pd.DataFrame({'datetimes':Wgroup_copy['datetimes'],'W':Wgroup_copy['new_value']})
    E = pd.DataFrame({'datetimes':Egroup_copy['datetimes'],'E':Egroup_copy['new_value']})

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

###############################################################################################################################
# Remove vars if gt error threshold 
condition = abs(E['E']) > error_threshold 
U['Unew'] = np.where(condition == True, np.nan, U['U'])
V['Vnew'] = np.where(condition == True, np.nan, V['V'])
W['Wnew'] = np.where(condition == True, np.nan, W['W'])
E['Enew'] = np.where(condition == True, np.nan, E['E'])

###############################################################################################################################
#grid data 
print('gridding data ...')
if loco == 'mfd':
    Zcenter = np.arange(-80,-9,5)  # originally went to 
    binedges = Zcenter-2.5
    binedges = np.append(binedges,binedges[-1]+5)

if loco == 'nsif':
    Zcenter = np.arange(-50,-14,5)  # originally went to 
    binedges = Zcenter-2.5
    binedges = np.append(binedges,binedges[-1]+5)

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
# update: put UVWE new if filter before hand 
Zgroupi = Z.groupby('datetimes')['Z'].apply(list).reset_index(name='z')
Ugroupi = U.groupby('datetimes')['Unew'].apply(list).reset_index(name='u')
Vgroupi = V.groupby('datetimes')['Vnew'].apply(list).reset_index(name='v')
Wgroupi = W.groupby('datetimes')['Wnew'].apply(list).reset_index(name='w')
Egroupi = E.groupby('datetimes')['Enew'].apply(list).reset_index(name='e')
'''

for idx in range(NTc):
    ui = U['Unew'][U['datetimes']==dti[idx]]
    vi = V['Vnew'][V['datetimes']==dti[idx]]
    wi = W['Wnew'][W['datetimes']==dti[idx]]
    ei = E['Enew'][E['datetimes']==dti[idx]]
    zi = Z['Z'][Z['datetimes']==dti[idx]]

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
if np.all(Zgroupi['datetimes'].values==Ugroupi['datetimes'].values):
    ADCP['ocean_time'] = (('ocean_time'), Zgroupi['datetimes'].values, {'long_name':'datetimes from OOI, 01-JAN-1970 00:00:00'})
else: 
    print('exited script; werid time errors')
    sys.exit()

ADCP['z'] = (('ocean_time','z'), zb, {'units':'m', 'long_name':'altitude from OOI', 'positive':'up'})

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
    print('nsif: ' + loco)
    fn_o = out_dir + '/' + str(moor) + '_nsif_ADCP.nc'

if loco == 'mfd':
    ADCP['u'].attrs['moored_location'] = 'mfd, depth ~540m'
    ADCP['v'].attrs['moored_location'] = 'mfd, depth ~540m'
    ADCP['w'].attrs['moored_location'] = 'mfd, depth ~540m'
    print('mfd: ' + loco)
    fn_o = out_dir + '/' + str(moor) + '_mfd_ADCP.nc'

ADCP.to_netcdf(fn_o, unlimited_dims='ocean_time')

print('saved!')