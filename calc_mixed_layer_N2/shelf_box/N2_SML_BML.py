"""
Code calculates SA and SIG0 + N2
Locates position of N2 max 
Then calculates SML and BML using temp. threshold method 
Flag first instance where:
del T(z_rho) = T(z_rho) - T0 > -0.05 
where T0 is either the surface(bottom)-most temperature value

this value of z_rho is saved, and the thickness of the SML(BML) 
is estimated using the edge z_w value just below the flagged z_rho 

Note: this code was writted to work on nc-files extracted using 
LO/extract/box/extract_box_chunks.py
which have grid info saved. If re-write can get grid info using
a history file and G, S, T = zrfun.get_basic_info(), which can then 
calc z_rho and z_w 

We are exploring the shelf_box job here 
Want to calculate the SML BML and location of max N2 
for cas7_t0_x4b years 2014 - 2019 
(2019 = last full year processed; 2013 skipped for now)

Todos: write flags so can enter 'box' or 'moor' and the filename etc 
so can use command line 

"""

import xarray as xr
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

fnn = 'shelf_box_2014.01.01_2015.12.31'
#fnn = 'shelf_box_2016.01.01_2017.12.31'
#fnn = 'shelf_box_2018.01.01_2019.12.31'
fn_in = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'box' / (fnn + '_chunks') / (fnn + '.nc')

out_dir = Ldir['parent'] / 'LKH_output'/ 'calc_mixed_layer_N2' / 'shelf_box' / fnn
temp_dir = out_dir / ('temp_' + fnn)
Lfun.make_dir(out_dir, clean=True)
Lfun.make_dir(temp_dir, clean=True)

testing = True
    
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
g = 9.81
# po = 1024
po = round(np.nanmean(SIG0)+1000,0)
    
dp = np.diff(SIG0,axis=1)
dz = np.diff(z_rho,axis=1)
    
dpdz = dp/dz
N2 = -(g/po)*dpdz

C = np.argmax(N2,axis=1,keepdims=True)
z_N2max = np.take_along_axis(z_w.values,C+1,axis=1)
val_N2max = np.take_along_axis(N2,C,axis=1)

print('Time to calc N2 = %0.2f sec' % (time()-tt0))
sys.stdout.flush()
 
# 4. Calc SML using temp threshold 
tt0 = time()       
CTf = np.flip(CT,axis=1)             # flip s/t data sits surface:bottom  
z_wf = np.flip(z_w,axis=1)
z_rhof = np.flip(z_rho,axis=1)
             
tt = CTf[:,0,:,:]                    # surface-most values 
t0s = tt[:,np.newaxis,:,:]

dT_s = np.round(CTf-t0s,2)           # thresholds
Tbool = dT_s>-0.05

# grab the first index where false 
# keep dims = True makes grabbing the z variables way easier 
A = np.argmax(Tbool==False,axis=1,keepdims = True)     # read across the depth layers 

Ti_zrho = np.take_along_axis(z_rhof.values,A-1,axis=1) # A grabs the first false, so go up one rho-point
Ti_zw = np.take_along_axis(z_wf.values,A,axis=1)       # and this is the z_w (assumed depth of SML using T thresh)
val_Tsml = np.take_along_axis(CTf,A-1,axis=1)

zz = zsurf.values[:,np.newaxis,:,:]                    # now calc the depth of the SML
D_sml = zz + np.abs(Ti_zrho) 

print('Time to calc SML for 5 timestamps = %0.2f sec' % (time()-tt0))
sys.stdout.flush()    


yplotting = False  
# 5: check SML + N2 plotting 
tt0 = time()
idx = [120,80]
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
    plt.gcf().axes[axnum].axhline(y = Ti_zw[-1,:,idx[0],idx[1]], color = 'blue', label = 'axvline - full height')
    plt.gcf().axes[axnum].plot(val_N2max[-1,:,idx[0],idx[1]],z_N2max[-1,:,idx[0],idx[1]],color='black',marker='*')
    plt.gcf().axes[axnum].set_title('N2')
    plt.gcf().axes[axnum].set_ylabel('depth m')
    
    axnum = 2 
    Tplot = plt.gcf().axes[axnum].plot(CT[-1,:,idx[0],idx[1]],z_rho[-1,:,idx[0],idx[1]],color='pink',marker='x')     
    plt.gcf().axes[axnum].plot(val_Tsml[-1,:,idx[0],idx[1]],Ti_zrho[-1,:,idx[0],idx[1]],color='black',marker='*')
    plt.gcf().axes[axnum].axhline(y = Ti_zw[-1,:,idx[0],idx[1]], color = 'r', label = 'axvline - full height')
    
    print('Time to plot data = %0.2f sec' % (time()-tt0))
    sys.stdout.flush() 

