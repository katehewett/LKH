"""
Code calculates SA and SIG0 + N2 and dT/dz
Locates position of N2 max and dT/dz max

Note: this code was writted to work on nc-files extracted using 
LO/extract/box/extract_box_chunks.py
which have grid info saved. If re-write can get grid info using
a history file and G, S, T = zrfun.get_basic_info(), which can then 
calc z_rho and z_w 

We are exploring the NHL_transect job here 
for cas7_t0_x4b years 2014 - 2019 
(2019 = last full year processed; 2013 skipped for now)

Todos: write flags so can enter 'box' or 'moor' and the filename etc 
so can use command line 

All years 
Time to load data = 0.09 sec
Time to extract data = 0.10 sec
Time to calc gsw vars = 72.00 sec
Time to calc N2 = 4.59 sec

"""

import xarray as xr
from xarray import open_dataset, Dataset
import numpy as np
import gsw
from lo_tools import Lfun, zrfun
from time import time
import sys
import gsw

#plotting things
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import cmcrameri.cm as cmc

Ldir = Lfun.Lstart()

fnn = 'NHL_transect_2014.01.01_2019.12.31'
fn_in = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'box' / (fnn + '_chunks') / (fnn + '.nc')

out_dir = Ldir['parent'] / 'LKH_output'/ 'calc_mixed_layer_N2' / 'NHL' / fnn
Lfun.make_dir(out_dir, clean=True)

testing = False
    
# 1. Get some grid information and calc SA + SIG0
tt0 = time()
ds = xr.open_dataset(fn_in, decode_times=False)
print('Time to load data = %0.2f sec' % (time()-tt0))

lat = ds.lat_rho
lon = ds.lon_rho
h = ds.h.values 
mask_rho = ds.mask_rho

z_w = ds.z_w
z_rho = ds.z_rho
tempC = ds.temp
SP = ds.salt

if testing:
    z_w = z_w[0:5,:,:,:]
    z_rho = z_rho[0:5,:,:,:]
    tempC = tempC[0:5,:,:,:]
    SP = SP[0:5,:,:,:]
    
zsurf = z_w[:,-1,:,:]
NT, NZ, NETA, NXI = tempC.shape
NW = z_w.shape[1]

print('Time to extract data = %0.2f sec' % (time()-tt0))
sys.stdout.flush()

# 3. Calc SA and SIG0: Need to loop over all timestamps NT 
# Reason: there's a holdup with sending gsw LARGE files - 
# it can accomodate 4D variables, but has an upper limit 
# And slow/can't run one of these shelf-box nc-files with all NT's using 
# personal computer
tt0 = time()
SA = np.nan * np.ones(SP.shape)           # initialize to hold results (time, z, x, y)
CT = np.nan * np.ones(SP.shape)
SIG0 = np.nan * np.ones(SP.shape)

for ii in range(NT):
    # run thru GSW for SA and SIG0
    iP = gsw.p_from_z(z_rho[ii,:,:,:],lat)
    iSA = gsw.SA_from_SP(SP[ii,:,:,:], iP, lon, lat)
    iCT = gsw.CT_from_pt(iSA, tempC[ii,:,:,:])
    iSIG0 = gsw.sigma0(iSA,iCT)   
    SA[ii,:,:,:] = iSA
    CT[ii,:,:,:] = iCT
    SIG0[ii,:,:,:] = iSIG0

print('Time to calc gsw vars = %0.2f sec' % (time()-tt0))
sys.stdout.flush()

# 4. calc N2 and find max N2 position
tt0 = time()
g = 9.8     
po = 1024
#po = round(np.nanmean(SIG0)+1000,0)
    
dp = np.diff(SIG0,axis=1)
dz = np.diff(z_rho,axis=1)
    
dpdz = dp/dz
N2 = -(g/po)*dpdz     # radians / second

C = np.argmax(N2,axis=1,keepdims=True)
z_N2max = np.take_along_axis(z_w.values,C+1,axis=1)
val_N2max = np.take_along_axis(N2,C,axis=1)

n = np.nan * np.ones(z_w.shape) 
n[:,1:NW-1,:,:] = N2

print('Time to calc N2 = %0.2f sec' % (time()-tt0))
sys.stdout.flush()

# 4. calc dT du dv dz's : u and v are on rho points
# plus shear and Rig
tt0 = time()
du = np.diff(ds.u.values,axis=1)
dv = np.diff(ds.v.values,axis=1)
dT = np.diff(CT,axis=1)
dS = np.diff(SA,axis=1)
dudz = du/dz
dvdz = dv/dz
dtdz = dT/dz

S2 = np.square(dudz) + np.square(dvdz)
zw = ds.z_w.values[:,1:NW-1,:,:]

s = np.nan * np.ones(z_w.shape) 
s[:,1:NW-1,:,:] = S2

#Rig = N2/S2
#Rig_N = np.log10(Rig/0.25)

print('Time to calc dzs and S2 = %0.2f sec' % (time()-tt0))
sys.stdout.flush()
 
#5. put in a dataset 
dsave = True
if dsave:     
    ds1 = Dataset()

    ds1['ocean_time'] = ds.ocean_time
    ds1['z_rho'] = ds.z_rho
    ds1['z_w'] = ds.z_w
    ds1['h'] = ds.h
    ds1['mask_rho'] = ds.mask_rho
    ds1['lat'] = ds.lat_rho
    ds1['lon'] = ds.lon_rho

    ds1['SIG0'] = (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),SIG0,{'units':'kg/m3 - 1000','long_name':'potential density anomaly'})
    ds1['SA'] = (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),SA,{'units':'g/kg','long_name':'Absolute Salinity'})
    ds1['CT'] = (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),CT,{'units':'degrees C','long_name':'Conservative Temperature'})

    ds1['N2'] = (('ocean_time', 's_w', 'eta_rho', 'xi_rho'),n,{'units':'radians/s2','long_name':'buoyancy frequency'})
    ds1['N2_max'] = (('ocean_time', 'eta_rho', 'xi_rho'),val_N2max.squeeze(),{'units':'radians/s2','long_name':'value max buoyancy frequency'})
    ds1['zN2_max'] = (('ocean_time', 'eta_rho', 'xi_rho'),z_N2max.squeeze(),{'units':'radians/s2','long_name':'location of max buoyancy frequency'})

    ds1['S2'] = (('ocean_time', 's_w', 'eta_rho', 'xi_rho'),s,{'units':'s^-2','long_name':'vertical shear squared'})
    
    fn_s = fnn + '_N2_S2.nc'
    this_fn = out_dir / (fn_s)
    ds1.to_netcdf(this_fn)
    
yplotting = False  
# 5: check SML + N2 plotting 
tt0 = time()
#idx = [1,50]
idx = [1,70]
if yplotting==True:
    # initialize plot 
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))

    ax1 = plt.subplot2grid((1,4),(0,0),colspan=1,rowspan=1)
    ax2 = plt.subplot2grid((1,4),(0,1),colspan=1,rowspan=1)
    ax3 = plt.subplot2grid((1,4),(0,2),colspan=1,rowspan=1)
    ax4 = plt.subplot2grid((1,4),(0,3),colspan=1,rowspan=1)
    
    axnum = 0 
    sigplot = plt.gcf().axes[axnum].plot(SIG0[-1,:,idx[0],idx[1]],z_rho[-1,:,idx[0],idx[1]],color='pink',marker='x') 
    plt.gcf().axes[axnum].set_title('SIG0')
    plt.gcf().axes[axnum].set_ylabel('depth m')
       
    axnum = 1 
    N2plot = plt.gcf().axes[axnum].plot(N2[-1,:,idx[0],idx[1]],z_w.values[-1,1:-1,idx[0],idx[1]],color='pink',marker='x') 
    plt.gcf().axes[axnum].plot(n[-1,:,idx[0],idx[1]],z_w.values[-1,:,idx[0],idx[1]],color='blue',marker='x') 
    plt.gcf().axes[axnum].plot(val_N2max[-1,:,idx[0],idx[1]],z_N2max[-1,:,idx[0],idx[1]],color='black',marker='*')
    plt.gcf().axes[axnum].set_title('N2')
    
    axnum = 2 
    Tplot = plt.gcf().axes[axnum].plot(S2[-1,:,idx[0],idx[1]],zw[-1,:,idx[0],idx[1]],color='pink',marker='x')
    plt.gcf().axes[axnum].set_title('S2')
    
    axnum = 3 
    Tplot = plt.gcf().axes[axnum].plot(R,zw[-1,:,idx[0],idx[1]],color='pink',marker='x')
    plt.gcf().axes[axnum].set_title('Rig')
    
    print('Time to plot data = %0.2f sec' % (time()-tt0))
    sys.stdout.flush() 

