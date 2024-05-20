"""
Code calculates SML and BML using temp. threshold method 
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

# 3. loop over all timestamps NT (reason: there's a holdup with sending 
# gsw LARGE files - it can accomodate 4D variables, but has an upper limit 
# And slow/can't run one of these shelf-box nc-files with all NT's using 
# personal computer) 
SA = np.nan * np.ones(SP.shape)           # initialize to hold results (time, z, x, y)
CT = np.nan * np.ones(SP.shape)
SIG0 = np.nan * np.ones(SP.shape)
P = np.nan * np.ones(SP.shape)
Tsml_zrho = np.nan * np.ones(SP.shape)
Tsml_zw = np.nan * np.ones(SP.shape)
Ssml_zrho = np.nan * np.ones(SP.shape)
Ssml_zw = np.nan * np.ones(SP.shape)

tt0 = time()
proc_list = []
for ii in range(NT):
    # run thru GSW for SA and SIG0
    iP = gsw.p_from_z(z_rho[ii,:,:,:],lat)
    iSA = gsw.SA_from_SP(SP[ii,:,:,:], iP, lon, lat)
    iCT = gsw.CT_from_pt(iSA, tempC[ii,:,:,:])
    iSIG0 = gsw.sigma0(iSA,iCT)
    
    # CALC SML with sig and temp thresholds
    SIGf = np.flipud(iSIG0.squeeze())
    Tf = np.flipud(iCT.squeeze())
    #dzrf = np.flipud(dzr)
    z_wf = np.flipud(z_w[ii,:,:,:].squeeze())
    z_rhof = np.flipud(z_rho[ii,:,:,:].squeeze())
    
    sig0s = SIGf[0,:,:]                      
    t0s = Tf[0,:,:]

    dSIG_s = np.round(SIGf-sig0s,2)
    dT_s = np.round(Tf-t0s,2)

    Tbool = dT_s>-0.05
    Sbool = dSIG_s<0.02
    
    # grab the first index where false 
    # keep dims = True makes grabbing the z variables way easier 
    A = np.argmax(Tbool==False,axis=0,keepdims = True)    
    B = np.argmax(Sbool==False,axis=0,keepdims = True)
    
    Ti_zrho = np.take_along_axis(z_rhof,A,axis=0)
    Aw = A+1
    Ti_zw = np.take_along_axis(z_wf,Aw,axis=0)
    
    Si_zrho = np.take_along_axis(z_rhof,B,axis=0)
    Bw = B+1
    Si_zw = np.take_along_axis(z_wf,Bw,axis=0)
    
    Tsml_zrho[ii,:,:,:] = Ti_zrho
    Tsml_zw[ii,:,:,:] = Ti_zw
    Ssml_zrho[ii,:,:,:] = Si_zrho
    Ssml_zw[ii,:,:,:] = Si_zw
    
    SA[ii,:,:,:] = iSA
    CT[ii,:,:,:] = iCT
    SIG0[ii,:,:,:] = iSIG0

print('Time to calc 5 data = %0.2f sec' % (time()-tt0))
sys.stdout.flush()    

tt0 = time()
yplotting = True 
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
    Tplot = plt.gcf().axes[axnum].plot(CT[-1,:,150,150].squeeze(),z_rho[-1,:,150,150].squeeze(),color='pink',marker='x') 
    plt.gcf().axes[axnum].plot(CT[-1,A[0,150,150],150,150].squeeze(),z_rhof[A[0,150,150],150,150],color='black',marker='*')
    plt.gcf().axes[axnum].plot(CT[-1,A[0,150,150],150,150].squeeze(),z_wf[A[0,150,150]+1,150,150],color='red',marker='s')
    plt.gcf().axes[axnum].set_title('CT')
    plt.gcf().axes[axnum].set_ylabel('depth m')
    
    axnum = 2
    rhoplot = plt.gcf().axes[axnum].plot(SIG0[-1,:,150,150].squeeze(),z_rho[-1,:,150,150].squeeze(),color='pink',marker='x') 
    plt.gcf().axes[axnum].set_title('SIG0')
    
    print('Time to plot data = %0.2f sec' % (time()-tt0))
    sys.stdout.flush()  

