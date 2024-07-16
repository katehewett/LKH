'''
plotting seasonal maps of 
sml 
N2max + depth 



'''

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys 
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cmcrameri import cm as cm2
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd


tt0 = time()

Ldir = Lfun.Lstart()

fn_o = Ldir['parent'] / 'plotting' / 'clines' 
fn_i = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'clines' / 'shelf_box_2014.01.01_2014.12.31' 
fn_in = fn_i / 'shelf_box_pycnocline_2014.01.01_2014.12.31.nc'

ds = xr.open_dataset(fn_in, decode_times=True)     

ot = pd.to_datetime(ds.ocean_time.values)

xrho = ds['lon_rho'].values
yrho = ds['lat_rho'].values
mask_rho = ds['mask_rho'].values
h = ds['h'].values

#var = ds['zSML'].values
var = ds['zDGmax'].values

mmonth = ot.month
myear = ot.year

# sort by season
dx1 = np.where((mmonth>=1) & (mmonth<=2)); 
dx2 = np.where((mmonth>=3) & (mmonth<=4)); 
dx3 = np.where((mmonth>=5) & (mmonth<=6)); 
dx4 = np.where((mmonth>=7) & (mmonth<=8)); 
dx5 = np.where((mmonth>=9) & (mmonth<=10)); 
dx6 = np.where((mmonth>=11) & (mmonth<=12)); 

winter = {}
spring = {}
summer = {}
fall = {}

odx1 = var[dx1]                                   
odx2 = var[dx2]
odx3 = var[dx3]
odx4 = var[dx4] 
odx5 = var[dx5] 
odx6 = var[dx6] 

masked_data = np.ma.masked_array(odx1, np.isnan(odx1)) 
O1 = np.ma.average(masked_data, axis=0,keepdims=False)  
O1[mask_rho==0] = np.nan 

masked_data = np.ma.masked_array(odx2, np.isnan(odx2)) 
O2 = np.ma.average(masked_data, axis=0,keepdims=False)  
O2[mask_rho==0] = np.nan 

masked_data = np.ma.masked_array(odx3, np.isnan(odx3)) 
O3 = np.ma.average(masked_data, axis=0,keepdims=False)  
O3[mask_rho==0] = np.nan 

masked_data = np.ma.masked_array(odx4, np.isnan(odx4)) 
O4 = np.ma.average(masked_data, axis=0,keepdims=False)  
O4[mask_rho==0] = np.nan 

masked_data = np.ma.masked_array(odx5, np.isnan(odx5)) 
O5 = np.ma.average(masked_data, axis=0,keepdims=False)  
O5[mask_rho==0] = np.nan 

masked_data = np.ma.masked_array(odx6, np.isnan(odx6)) 
O6 = np.ma.average(masked_data, axis=0,keepdims=False)  
O6[mask_rho==0] = np.nan 
 
#winter['var_mean'] = OW
#spring['var_mean'] = OSp
#summer['var_mean'] = OSu
#fall['var_mean'] = OF

# make fig
plt.close('all')
fs=10
plt.rc('font', size=fs)
fig = plt.figure(figsize=(11,8))
fig.set_size_inches(11,8, forward=False)
fig.tight_layout()
 
ax1 = plt.subplot2grid((1,6),(0,0),colspan=1,rowspan=1)
ax2 = plt.subplot2grid((1,6),(0,1),colspan=1,rowspan=1)
ax3 = plt.subplot2grid((1,6),(0,2),colspan=1,rowspan=1)

ax4 = plt.subplot2grid((1,6),(0,3),colspan=1,rowspan=1)
ax5 = plt.subplot2grid((1,6),(0,4),colspan=1,rowspan=1)
ax6 = plt.subplot2grid((1,6),(0,5),colspan=1,rowspan=1)

smin = 0 
smax = -80
# map
pfun.add_coast(ax1)
pfun.dar(ax1)
ax1.axis([-128, -123, 43, 50])
cp1 = ax1.pcolormesh(xrho, yrho, O1 ,cmap=cm2.roma_r) #, vmax = smin, vmin = smax)
ax1.contour(xrho,yrho,h, [80, 200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax1.set_xticks([-128, -126, -124])
ax1.set_title('Jan-Feb')
ax1.set_xlabel('zDGmax 2014')
#axw.grid(True)
plt.colorbar(cp1, orientation = "horizontal")

pfun.add_coast(ax2)
pfun.dar(ax2)
ax2.axis([-128, -123, 43, 50])
cp2 = ax2.pcolormesh(xrho, yrho, O2 ,cmap=cm2.roma_r) #, vmax = smin, vmin = smax)
ax2.contour(xrho,yrho,h, [80, 200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax2.set_xticks([-128, -126, -124])
ax2.set_title('Mar-Apr')
ax2.set_yticklabels([])
plt.colorbar(cp2, orientation = "horizontal")

pfun.add_coast(ax3)
pfun.dar(ax3)
ax3.axis([-128, -123, 43, 50])
cp3 = ax3.pcolormesh(xrho, yrho, O3 ,cmap=cm2.roma_r) #, vmax = smin, vmin = smax)
ax3.contour(xrho,yrho,h, [80, 200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax3.set_xticks([-128, -126, -124])
ax3.set_title('May-Jun')
ax3.set_yticklabels([])
plt.colorbar(cp3, orientation = "horizontal")

pfun.add_coast(ax4)
pfun.dar(ax4)
ax4.axis([-128, -123, 43, 50])
cp4 = ax4.pcolormesh(xrho, yrho, O4 ,cmap=cm2.roma_r) #, vmax = smin, vmin = smax)
ax4.contour(xrho,yrho,h, [80, 200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax4.set_xticks([-128, -126, -124])
ax4.set_title('Jul-Aug')
ax4.set_yticklabels([])
plt.colorbar(cp4, orientation = "horizontal")

pfun.add_coast(ax5)
pfun.dar(ax5)
ax5.axis([-128, -123, 43, 50])
cp5 = ax5.pcolormesh(xrho, yrho, O5 ,cmap=cm2.roma_r) #, vmax = smin, vmin = smax)
ax5.contour(xrho,yrho,h, [80, 200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax5.set_xticks([-128, -126, -124])
ax5.set_title('Sep-Oct')
ax5.set_yticklabels([])
plt.colorbar(cp5, orientation = "horizontal")

pfun.add_coast(ax6)
pfun.dar(ax6)
ax6.axis([-128, -123, 43, 50])
cp6 = ax6.pcolormesh(xrho, yrho, O6 ,cmap=cm2.roma_r) #, vmax = smin, vmin = smax)
ax6.contour(xrho,yrho,h, [80, 200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax6.set_xticks([-128, -126, -124])
ax6.set_title('Nov-Dec')
ax6.yaxis.set_label_position("right")
ax6.yaxis.tick_right()
plt.colorbar(cp6, orientation = "horizontal")

sys.exit()

