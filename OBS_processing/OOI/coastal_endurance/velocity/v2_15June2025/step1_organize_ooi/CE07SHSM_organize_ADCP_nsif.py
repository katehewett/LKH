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
Altered to speed up post 8JUne2025
'''

import sys
import os
import xarray as xr
import numpy as np
import pandas as pd
import posixpath 
from datetime import datetime
from scipy import stats
from scipy import interpolate
import warnings

import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart()

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
###############################################################################################################################

error_threshold = 0.1

moor = 'CE07SHSM'
#loco = 'mfd'
loco = 'nsif'

in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco 

if loco == 'nsif':
    fn_in = posixpath.join(in_dir, 'ooi-ce07shsm-rid26-01-adcpta000_dad4_820b_2d26.nc') 

if loco == 'mfd':
    fn_in = posixpath.join(in_dir, 'ooi-ce07shsm-mfd35-04-adcptc000_dad4_820b_2d26.nc') 

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

###############################################################################################################################
###############################################################################################################################
# OOI download provides data in a long 1D array 
df = pd.DataFrame({'datetimes':ds.time.values})
df['date'] = df['datetimes'].dt.date

df['z'] = ds.z.values
df['u'] = ds.eastward_sea_water_velocity.values
df['v'] = ds.northward_sea_water_velocity.values
#df['w'] = ds.upward_sea_water_velocity.values
df['e'] = ds.velprof_evl.values

#################################################################################################
# set vars to nan if > error threshold 
print('filtering error terms')
df['Econdition'] = abs(df['e']) > error_threshold 
df.loc[df['Econdition'],'u'] = np.nan # take all the u rows where Econdition == True and set to nan
df.loc[df['Econdition'],'v'] = np.nan
#df.loc[df['Econdition'],'w'] = np.nan # error is for u,v, but flagging w. not sure what's best here...
df.loc[df['Econdition'],'e'] = np.nan

#################################################################################################
# Want to drop some data here to make the arrays smaller:
# (1) weird OOI flags had a few timestamps with entire WC was shifted upwards. 
# clipping everything above -6m would remove those values here (rows dropped = 631) 
# That clip (-6m) also removed the duplicate zgroups per timestamp for CE07SHSM nsif
# (2) Since we're later going to clip at a surfacemost (and bottommost) binedge(s),
# we skipped a step and jumped to clip below -7.5m (instead of -6m) binedges[-1] and 
# -57.5m,binedges[0], which removes the deep (below -89m bins) present early in the timeseries
Zcenter = np.arange(-55,-9,5)  # originally went to 
binedges = Zcenter-2.5
binedges = np.append(binedges,binedges[-1]+5)

df.drop(df.loc[df['z']>binedges[-1]].index,inplace=True)
df.drop(df.loc[df['z']<binedges[0]].index,inplace=True)

df = df.reset_index() # reset index after drops

# Calc U outliers 
print('calc u outliers...')
Umean, bin_edges, binnumber = stats.binned_statistic(df['z'], df['u'], statistic=np.nanmean, bins=binedges)
Ustd, bin_edges, binnumber = stats.binned_statistic(df['z'], df['u'], statistic=np.nanstd, bins=binedges)
uhigh = Umean+7*Ustd
ulow = Umean-7*Ustd
# adjust bin numbers to python 0 = 1st and assign uhigh and ulows
Ubin = binnumber-1
UH = uhigh[Ubin]
UL = ulow[Ubin]
Ucondition = (df['u']<UL) | (df['u']>UH)
#np.shape(np.where(Ucondition==True))[1]

# Calc V outliers 
print('calc v outliers...')
Vmean, bin_edges, binnumber = stats.binned_statistic(df['z'], df['v'], statistic=np.nanmean, bins=binedges)
Vstd, bin_edges, binnumber = stats.binned_statistic(df['z'], df['v'], statistic=np.nanstd, bins=binedges)
vhigh = Vmean+7*Vstd
vlow = Vmean-7*Vstd
# adjust bin numbers to python 0 = 1st and assign uhigh and ulows
Vbin = binnumber-1
VH = vhigh[Vbin]
VL = vlow[Vbin]
Vcondition = (df['v']<VL) | (df['v']>VH)

'''
# Calc W outliers 
print('calc w outliers...')
Wmean, bin_edges, binnumber = stats.binned_statistic(df['z'], df['w'], statistic=np.nanmean, bins=binedges)
Wstd, bin_edges, binnumber = stats.binned_statistic(df['z'], df['w'], statistic=np.nanstd, bins=binedges)
whigh = Wmean+7*Wstd
wlow = Wmean-7*Wstd
# adjust bin numbers to python 0 = 1st and assign uhigh and ulows
Wbin = binnumber-1
WH = whigh[Wbin]
WL = wlow[Wbin]
Wcondition = (df['w']<WL) | (df['w']>WH)
'''

print('setting outliers to nan...')
df['UVcondition'] = Ucondition | Vcondition
#df['Wcondition'] = Wcondition

# if flag in u,v then uvw = nan
df.loc[df['UVcondition'],'u'] = np.nan
df.loc[df['UVcondition'],'v'] = np.nan
df.loc[df['UVcondition'],'w'] = np.nan
# flag outliers in w
#df.loc[df['Wcondition'],'w'] = np.nan

# 'mean' goes WAY WAY faster than np.nan in binned.statistic. So will drop nan's from df here before grouping:
df2 = df.copy()
#columns_to_drop = ['UVcondition', 'Econdition', 'Wcondition','index']
columns_to_drop = ['UVcondition', 'Econdition', 'index']
df2 = df2.drop(columns_to_drop,axis=1)
NU=np.where(np.isnan(df2['u']))
NV=np.where(np.isnan(df2['v']))
if np.all(NU[0]==NV[0]): 
    print('okay')
df2.drop(index=NU[0],axis=0,inplace=True) #doens't return a new df
df2 = df2.reset_index()
columns_to_drop = ['index']
df2 = df2.drop(columns_to_drop,axis=1)

'''
NW = np.where(np.isnan(df2['w']))
columns_to_drop = ['u', 'v', 'e','index']
dfw = df2.drop(index=NW[0],axis=0,inplace=False) # returns a new df, named dfw
dfw.reset_index(inplace=True)
dfw = dfw.drop(columns_to_drop,axis=1)
'''

###############################################################################################################################
###############################################################################################################################
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

Ugroup = df.groupby('datetimes')['u'].apply(list).reset_index(name='u')
print('grouped u')
Vgroup = df.groupby('datetimes')['v'].apply(list).reset_index(name='v')
print('grouped v')
Egroup = df.groupby('datetimes')['e'].apply(list).reset_index(name='e')
print('grouped error')

'''
Zwgroup = dfw.groupby('datetimes')['z'].apply(list).reset_index(name='z') 
Wgroup = dfw.groupby('datetimes')['w'].apply(list).reset_index(name='w')
print('grouped w')
'''

###############################################################################################################################
'''
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


sys.exit()
'''
###############################################################################################################################
#grid data 
print('gridding data ...')
if loco == 'mfd':
    Zcenter = np.arange(-85,-24,1)  # originally went to 
    binedges = Zcenter-2.5
    binedges = np.append(binedges,binedges[-1]+5)

if loco == 'nsif':
    Zcenter = np.arange(-55,-7,1)  
    binedges = Zcenter-0.5
    binedges = np.append(binedges,binedges[-1]+1)

NZc = np.shape(Zcenter)[0]
NTc = np.shape(Zgroup['datetimes'].unique())[0]

#dti = Zgroup['datetimes'].unique()
print('NTc: '+str(NTc))

zb = np.ones([NTc,NZc])*np.nan
ub = np.ones([NTc,NZc])*np.nan
vb = np.ones([NTc,NZc])*np.nan
#wb = np.ones([NTc,NZc])*np.nan
eb = np.ones([NTc,NZc])*np.nan

for idx in range(NTc):
    ui = Ugroup['u'][idx] 
    vi = Vgroup['v'][idx] 
    #wi = Wgroup['w'][idx] 
    ei = Egroup['e'][idx] 
    zi = Zgroup['z'][idx] 

    #statistic, bin_edges, binnumber = stats.binned_statistic(x, values, statistic='mean', bins=bins)
    #x the data to be binned; values the values on which the statistic will be computed 
    # values must have the same shape as x or be a list of arrays with the same shape as x)
    # x = [1, 1, 2, 5, 7] # values = [[1.0, 1.0, 2.0, 1.5, 3.0], [2.0, 2.0, 4.0, 3.0, 6.0]]
    # bins = [1, 4, 7] # statistic >> array([[1.33333333, 2.25],[2.66666667, 4.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ustat, bin_edges, binnumber = stats.binned_statistic(zi, ui, statistic='mean', bins=binedges)
        vstat, bin_edges, binnumber = stats.binned_statistic(zi, vi, statistic='mean', bins=binedges)
        #wstat, bin_edges, binnumber = stats.binned_statistic(zi, wi, statistic='mean', bins=binedges)
        estat, bin_edges, binnumber = stats.binned_statistic(zi, ei, statistic='mean', bins=binedges)

    zb[idx,:] = Zcenter # just a checker
    ub[idx,:] = ustat
    vb[idx,:] = vstat
    #wb[idx,:] = wstat
    eb[idx,:] = estat

    #plt.plot(ui,zi,'bo-')
    #plt.plot(ub[idx,:],zb[idx,:],'r*')

###############################################################################################################################
# put to ds and save 
print('putting to ds...')

ADCP = xr.Dataset()
if np.all(Zgroup['datetimes'].values==Ugroup['datetimes'].values):
    ADCP['ocean_time'] = (('ocean_time'), Zgroup['datetimes'].values, {'long_name':'datetimes from OOI, 01-JAN-1970 00:00:00'})
else: 
    print('exited script; werid time errors')
    sys.exit()

ADCP['z'] = (('ocean_time','z'), zb, {'units':'m', 'long_name':'altitude from OOI', 'positive':'up'})

ADCP['u'] = (('ocean_time','z'), ub, {'units':'m.s-1', 'long_name': 'OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'})
ADCP['v'] = (('ocean_time','z'), vb, {'units':'m.s-1', 'long_name': 'OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity'})
#ADCP['w'] = (('ocean_time','z'), wb, {'units':'m.s-1', 'long_name': 'Upward Sea Water Velocity','positive':'upward',
#                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity'})
ADCP['velprof'] = (('ocean_time','z'), eb, {'units':'m.s-1', 'long_name': 'Error Seawater Velocity detail link broken','positive':'unclear',
                                      'metadata link':'BROKEN: https://mmisw.org/ont/cf/parameter/VELPROF-EVL'})

if loco == 'nsif':
    ADCP['u'].attrs['moored_location'] = 'nsif ~7m below surface'
    ADCP['v'].attrs['moored_location'] = 'nsif ~7m below surface'
    #ADCP['w'].attrs['moored_location'] = 'nsif ~7m below surface'
    print('nsif: ' + loco)
    fn_o = posixpath.join(out_dir , str(moor) + '_nsif_ADCP.nc')

if loco == 'mfd':
    ADCP['u'].attrs['moored_location'] = 'mfd, depth ~540m'
    ADCP['v'].attrs['moored_location'] = 'mfd, depth ~540m'
    #ADCP['w'].attrs['moored_location'] = 'mfd, depth ~540m'
    print('mfd: ' + loco)
    fn_o = posixpath.join(out_dir , str(moor) + '_mfd_ADCP.nc')

ADCP.to_netcdf(fn_o, unlimited_dims='ocean_time')

print('saved!')





'''
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