"""
Step 2: 
code calculates N2 and dTdz
finds position of N2max and dTdzmax -
flags CT SA and SIG0 at those depths 
saves file

Note: this code was writted to work on 
data extracted from LO at the NHL_transect 
for cas7_t0_x4b years 2014 - 2019 
(2019 = last full year processed; 2013 skipped for now)

"""

import xarray as xr
from xarray import open_dataset, Dataset
import numpy as np
import gsw
from lo_tools import Lfun, zrfun
from time import time
import sys
import gsw
import math 

#plotting things
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import cmcrameri.cm as cmc

# when dealing with dates
from datetime import date, timedelta
import pandas as pd

Ldir = Lfun.Lstart()

fnn = 'NHL_transect_2014.01.01_2019.12.31'

in_dir = Ldir['parent'] / 'LKH_output'/ 'NHL' / 'explore_stratification' / fnn
fn_in = in_dir / (fnn + '_phys.nc')
    
testing = False

# 1. Get some grid information and calc SA + SIG0
tt0 = time()
ds = xr.open_dataset(fn_in, decode_times=False)

lat = ds.lat_rho
lon = ds.lon_rho
h = ds.h.values 
mask_rho = ds.mask_rho

z_w = ds.z_w
z_rho = ds.z_rho
CT = ds.CT
SA = ds.SA
SIG0 = ds.SIG0

if testing:
    z_w = z_w[200:215,:,:,:]        # take subset of NT at all depths
    z_rho = z_rho[200:215,:,:,:]
    CT = CT[200:215,:,:,:]
    SA = SA[200:215,:,:,:]
    SIG0 = SIG0[200:215,:,:,:]
    
zsurf = np.expand_dims(z_w[:,-1,:,:],axis=1)
z10 = zsurf - 10 
SIG10 = SIG0.where(z_rho>=z10)

NT, NZ, NR, NC = CT.shape

# the dates ocean_time are days since 2014 1 1 and are at 12 noon
# just using the dates and dropping time // fix this later
# not sure why not at 8pm (daily = 8pm; but low pass - 12noon)
start_date = date(2014,1,1)
end_date = date(2019,12,31)
delta = timedelta(days=1)
odate = pd.date_range(start_date,end_date,freq='D')

print('Time to load data = %0.2f sec' % (time()-tt0))
sys.stdout.flush()

# 1. calc N2 and find max N2 position
tt0 = time()
g = 9.8     
po = 1025 
#po = round(np.nanmean(SIG0)+1000,0) # = 1025 all years in this domain 
    
dp = np.diff(SIG0,axis=1)
dt = np.diff(CT,axis=1)
dz = np.diff(z_rho,axis=1)
    
dpdz = dp/dz
dtdz = dt/dz
N2 = -(g/po)*dpdz     # radians / second

C = np.argmax(N2,axis=1,keepdims=True)
z_N2max = np.take_along_axis(z_w.values,C+1,axis=1)
val_N2max = np.take_along_axis(N2,C,axis=1)

B = np.argmax(dtdz,axis=1,keepdims=True)
z_dztmax = np.take_along_axis(z_w.values,B+1,axis=1)
val_dztmax = np.take_along_axis(dtdz,B,axis=1)

# convert to cycles/hr
s = np.sign(val_N2max)
N = (abs(val_N2max)**(1/2))*s
Nhr = (N/(2*math.pi))*60*60

print('Time to calc N2 = %0.2f sec' % (time()-tt0))
sys.stdout.flush()

# PLOT surf T S then add N2-max depth (and value)
X = ds.lon_rho.values.squeeze()
Y = odate #ds.ocean_time.values
Nz = z_N2max.squeeze()
Nv = Nhr.squeeze()
dzTz = z_dztmax.squeeze()
S = SA[:,-1,:,:].squeeze()
T = CT[:,-1,:,:].squeeze()
S0 = SIG0[:,-1,:,:].squeeze()
mdates1 = pd.date_range(start='2014-01-01',end='2019-12-31', freq='6MS', inclusive = 'both')

plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(16,8))

ax1 = plt.subplot2grid((1,4),(0,0),colspan=1,rowspan=1)
ax2 = plt.subplot2grid((1,4),(0,1),colspan=1,rowspan=1)
ax3 = plt.subplot2grid((1,4),(0,2),colspan=1,rowspan=1)
ax4 = plt.subplot2grid((1,4),(0,3),colspan=1,rowspan=1)
    
# Create sig0 surf plot
siglevels = np.arange(21,25.1,0.1)
sigmap=cmc.roma_r.with_extremes(under='Navy',over='Maroon')

axnum = 0
isig = plt.gcf().axes[axnum].contourf(X,Y,S0,siglevels,cmap=sigmap,extend="both")
plt.gcf().axes[axnum].invert_yaxis()
plt.gcf().axes[axnum].set_title('LO/NHL_transect: SIG0')

scb = plt.gcf().colorbar(isig, ticks = [21,22,23,24,25], location='bottom', pad = 0.05, label='surface kg/m3 - 1000')

# Create N2 depth plot
#levels = np.arange(-260,0,5) # -260 : -5 spaced at 5  (min -257m)
a = np.arange(-260,-40,10)
b = np.arange(-40,0,5)
Nlevels = np.concatenate((a,b))

nmap=cmc.bukavu.with_extremes(over='white')

axnum = 1
# iN = plt.gcf().axes[axnum].contourf(X,Y,Nz, Nlevels,cmap = cmc.bukavu)
iN = plt.gcf().axes[axnum].contourf(X,Y,Nz,Nlevels,cmap=nmap,extend="max")
plt.gcf().axes[axnum].invert_yaxis()
plt.gcf().axes[axnum].set_title('depth N2 max')

Ncb = plt.gcf().colorbar(iN, ticks = [-260,-200,-130,-70,-30,-5], location='bottom', pad = 0.05, label='N2max depth m') 

plt.gcf().axes[axnum].set_yticks(mdates1)  
date_form = mdates.DateFormatter('%b')
plt.gcf().axes[axnum].yaxis.set_major_formatter(date_form)
#plt.gcf().axes[axnum].set_yticklabels([])


# Create N2 val plot
Nvlevels = np.arange(0,24,1)
nvmap=cmc.roma_r.with_extremes(under='Navy',over='Maroon')

axnum = 2
iNv = plt.gcf().axes[axnum].contourf(X,Y,Nv,Nvlevels,cmap=nvmap,extend="both")
plt.gcf().axes[axnum].invert_yaxis()
plt.gcf().axes[axnum].set_title('val N max')

Ncb = plt.gcf().colorbar(iNv, ticks = [0,2,4,6,8,10,14,18,24], location='bottom', pad = 0.05, label='N rad/hr') 

plt.gcf().axes[axnum].set_yticks(mdates1)  
date_form = mdates.DateFormatter('%b')
plt.gcf().axes[axnum].yaxis.set_major_formatter(date_form)
#plt.gcf().axes[axnum].set_yticklabels([])

# Create dzT depth plot
axnum = 3
idzT = plt.gcf().axes[axnum].contourf(X,Y,dzTz,Nlevels,cmap=nmap,extend="max")
plt.gcf().axes[axnum].invert_yaxis()
plt.gcf().axes[axnum].set_title('depth dzTz max')

dTcb = plt.gcf().colorbar(idzT, ticks = [-260,-200,-130,-70,-30,-5], location='bottom', pad = 0.05, label='dzT max depth m') 

plt.gcf().axes[axnum].set_yticks(mdates1)  
date_form = mdates.DateFormatter('%b')
plt.gcf().axes[axnum].yaxis.set_major_formatter(date_form)
#plt.gcf().axes[axnum].set_yticklabels([])
