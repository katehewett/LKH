'''
Following CE07SHSM
TODO: 
update readme and text 

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
nstdev = 7 

moor = 'CE07SHSM'
#loco = 'mfd'
loco = 'nsif'

in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco 

if loco == 'nsif':
    fn_in = posixpath.join(in_dir, 'ooi-ce07shsm-rid26-01-adcpta000_dad4_820b_2d26.nc') 

#if loco == 'mfd':
#    fn_in = posixpath.join(in_dir, 'ooi-ce09ossm-mfd35-04-adcpsj000_ff2d_359a_11eb.nc') 

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
'''# set vars to nan if > error threshold 
print('filtering error terms')
df['Econdition'] = abs(df['e']) > error_threshold 
df.loc[df['Econdition'],'u'] = np.nan # take all the u rows where Econdition == True and set to nan
df.loc[df['Econdition'],'v'] = np.nan
#df.loc[df['Econdition'],'w'] = np.nan # error is for u,v, but flagging w. not sure what's best here...
df.loc[df['Econdition'],'e'] = np.nan
'''
#################################################################################################
# dropping z values and outliers (see CE07shsm notes)
if loco == 'nsif':
    Zcenter = np.arange(-50,-14,5) 
    binedges = Zcenter-2.5
    binedges = np.append(binedges,binedges[-1]+5)

df.drop(df.loc[df['z']>binedges[-1]].index,inplace=True)
df.drop(df.loc[df['z']<binedges[0]].index,inplace=True)

df = df.reset_index() # reset index after drops

##################################################################################
# (1) calc stats to be used in threshold flags per 5m bin
print('calculating u & v thresholds ...')
Ucount, bin_edges, binnumber = stats.binned_statistic(df['z'], df['u'], 'count', bins=binedges)
# pause here to look at Ucount 

NZc = np.shape(Zcenter)[0]
b = binnumber-1
Umean = np.ones(NZc)*np.nan
Ustd = np.ones(NZc)*np.nan
Vmean = np.ones(NZc)*np.nan
Vstd = np.ones(NZc)*np.nan

for idx in range(NZc):
    ui = df['u'].iloc[b==idx]
    vi = df['v'].iloc[b==idx]

    Umean[idx] = np.nanmean(ui)
    Vmean[idx] = np.nanmean(vi)

    Ustd[idx] = np.nanstd(ui)
    Vstd[idx] = np.nanstd(vi)

uthresh = Umean+nstdev*Ustd
vthresh = Vmean+nstdev*Vstd

df['binnum_adj'] = b

df['uthresh']= np.copy(b).astype(float)
df['vthresh']= np.copy(b).astype(float)

for idx in range(0,NZc):
    df.loc[df['binnum_adj']==idx,'uthresh']=uthresh[idx].astype(float)
    df.loc[df['binnum_adj']==idx,'vthresh']=vthresh[idx].astype(float)

df['vthresh'] = df['vthresh'].round(1)
df['uthresh'] = df['uthresh'].round(1)

# (2) find where u (then v) exceed the threshold, and replace with nan
df['Unew'] = np.copy(df['u'])
df.loc[np.abs(df['u'])>df['uthresh'],'Unew']=np.nan

df['Vnew'] = np.copy(df['v']) 
df.loc[np.abs(df['v'])>df['vthresh'],'Vnew']=np.nan

# (2) find where |e| > error_threshold, and replace with nan
df.loc[np.abs(df['e'])>error_threshold,'Unew']=np.nan
df.loc[np.abs(df['e'])>error_threshold,'Vnew']=np.nan

# housekeeping: reset index after drop and remove idx columns
columns_to_drop = ['index']
df.drop(columns_to_drop, axis=1, inplace=True)

##################################################################################
# (1) Take daily average and then interpolate to Zcenter 
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

Zgrouped = df.groupby([pd.Grouper(freq='D'), 'binnum_adj'])['z'].mean().reset_index(name='z')
Ugrouped = df.groupby([pd.Grouper(freq='D'), 'binnum_adj'])['u'].mean().reset_index(name='u')
Vgrouped = df.groupby([pd.Grouper(freq='D'), 'binnum_adj'])['v'].mean().reset_index(name='v')
Egrouped = df.groupby([pd.Grouper(freq='D'), 'binnum_adj'])['v'].mean().reset_index(name='e')

'''
predrop = np.shape(df['z'])[0]

df.dropna(subset=['Unew','Vnew'],inplace = True)
postdrop = np.shape(df['z'])[0]

if (np.any(np.isnan(df['Unew'])) == True) | (np.any(np.isnan(df['Vnew']))==True):
    print('need to drop nans')
    sys.exit()
else: print('no nans in u/v')


print('dropped ' + str(predrop-postdrop) + ' rows after: uvthreshold & error threshold')
'''
##################################################################################
##################################################################################
#interp to Zcenter
print('interpolating velocities to Zcenter')
DT = np.unique(Zgrouped['date'])
NTc = np.shape(DT)[0]

Ui = np.ones([NTc,NZc])*np.nan
Vi = np.ones([NTc,NZc])*np.nan
Ei = np.ones([NTc,NZc])*np.nan

# this should be removed it's super slow. but whatever. TODO fix it later
print('grouping by timestamp...')
for idx in range(0,NTc):
    z = Zgrouped.loc[Zgrouped['date']==DT[idx],'z']
    u = Ugrouped.loc[Ugrouped['date']==DT[idx],'u']
    v = Vgrouped.loc[Vgrouped['date']==DT[idx],'v']
    e = Egrouped.loc[Egrouped['date']==DT[idx],'e']
    fu = interpolate.interp1d(z,u,bounds_error=False,fill_value = np.nan)
    fv = interpolate.interp1d(z,v,bounds_error=False,fill_value = np.nan)
    fe = interpolate.interp1d(z,e,bounds_error=False,fill_value = np.nan)

    Ui[idx,:] = fu(Zcenter)
    Vi[idx,:] = fv(Zcenter)
    Ei[idx,:] = fe(Zcenter)

    if idx % 100 == 0:
        print(f"Iteration: {idx}")

zz = np.expand_dims(Zcenter,axis=0)
zz = np.tile(zz,[NTc,1])

###############################################################################################################################
# put to ds and save 
print('putting to ds...')

ADCP = xr.Dataset()

ADCP['ocean_time'] = (('ocean_time'), DT, {'long_name':'datetimes from OOI, 01-JAN-1970 00:00:00'})

ADCP['z'] = (('ocean_time','z'), zz, {'units':'m', 'long_name':'altitude from OOI', 'positive':'up'})

ADCP['u'] = (('ocean_time','z'), Ui, {'units':'m.s-1', 'long_name': 'OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity'})
ADCP['v'] = (('ocean_time','z'), Vi, {'units':'m.s-1', 'long_name': 'OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity'})
#ADCP['w'] = (('ocean_time','z'), wb, {'units':'m.s-1', 'long_name': 'Upward Sea Water Velocity','positive':'upward',
#                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity'})
ADCP['e'] = (('ocean_time','z'), Ei, {'units':'m.s-1', 'long_name': 'Error Seawater Velocity detail link broken','positive':'unclear',
                                      'metadata link':'BROKEN: https://mmisw.org/ont/cf/parameter/VELPROF-EVL'})

if loco == 'nsif':
    ADCP['u'].attrs['moored_location'] = 'nsif ~7m below surface'
    ADCP['v'].attrs['moored_location'] = 'nsif ~7m below surface'
    #ADCP['w'].attrs['moored_location'] = 'nsif ~7m below surface'
    print('nsif: ' + loco)
    fn_o = posixpath.join(out_dir , str(moor) + '_nsif_ADCP_rev1.nc')

if loco == 'mfd':
    ADCP['u'].attrs['moored_location'] = 'mfd, depth ~87m'
    ADCP['v'].attrs['moored_location'] = 'mfd, depth ~87m'
    #ADCP['w'].attrs['moored_location'] = 'mfd, depth ~87m'
    print('mfd: ' + loco)
    fn_o = posixpath.join(out_dir , str(moor) + '_mfd_ADCP_rev1.nc')

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