'''
Plot ARAG < = 1 (0.5)  percent time bottom water calculated 
using step2_calc_percenttime_corrosive_seasonal.py

'''

# imports
import os 
import sys
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import xarray as xr
from time import time
import numpy as np

import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import cmcrameri.cm as cm
import pickle 

Ldir = Lfun.Lstart()

# add this in as a command line arguments
threshold = 1

fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'bottom_maps' / 'percent_time' 
pn = 'seasonal_percent_time_ARAG'+str(threshold)+'_bottom_OA_indicators.pkl'
picklepath = fn_o/pn

with open(picklepath, 'rb') as fp3:
    P = pickle.load(fp3)
    print('loaded file')

# load the shelf mask / assign vars 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf 
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

# PLOTTING
# map
    
plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

xmin = -125.5 
xmax = -123
xxticks = [-125.5, -124.5, -123.5]
xxticklabels = ['-125.5', '-124.5', '-123.5']

# WINTER!!!!! 
axw = plt.subplot2grid((3,5), (0,0), colspan=1,rowspan=3)
pfun.add_coast(axw)
pfun.dar(axw)
axw.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
axw.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axw.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
axw.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')

axw.set_title('Jan - Mar')
axw.set_xlabel('Longitude')
axw.set_ylabel('Latitude')
axw.set_xlim([xmin, xmax])
axw.set_ylim([42.75,48.75])
axw.set_xticks(xxticks)
axw.set_yticks([42.75,43,44,45,46,47,48,48.75])
axw.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axw.xaxis.set_ticklabels(xxticklabels)
axw.grid(False)

# SPRING!!!!! 
axs = plt.subplot2grid((3,5), (0,1), colspan=1,rowspan=3)
pfun.add_coast(axs)
pfun.dar(axs)
axs.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
axs.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axs.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
axs.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')

axs.set_title('Apr - Jun')
axs.set_xlabel('Longitude')
axs.set_xlim([xmin, xmax])
axs.set_ylim([42.75,48.75])
axs.set_xticks(xxticks)
axs.set_yticks([42.75,43,44,45,46,47,48,48.75])
axs.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axs.xaxis.set_ticklabels(xxticklabels)
axs.grid(False)

# Summer!!!!! 
axsu = plt.subplot2grid((3,5), (0,2), colspan=1,rowspan=3)
pfun.add_coast(axsu)
pfun.dar(axsu)
axsu.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
axsu.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axsu.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
axsu.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')

axsu.set_title('Jul - Sep')
axsu.set_xlabel('Longitude')
axsu.set_xlim([xmin, xmax])
axsu.set_ylim([42.75,48.75])
axsu.set_xticks(xxticks)
axsu.set_yticks([42.75,43,44,45,46,47,48,48.75])
axsu.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axsu.xaxis.set_ticklabels(xxticklabels)
axsu.grid(False)

# Fall!!!!! 
axf = plt.subplot2grid((3,5), (0,3), colspan=1,rowspan=3)
pfun.add_coast(axf)
pfun.dar(axf)
axf.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
axf.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axf.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
axf.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')

axf.set_title('Oct - Dec')
axf.set_xlabel('Longitude')
axf.set_xlim([xmin, xmax])
axf.set_ylim([42.75,48.75])
axf.set_xticks(xxticks)
axf.set_yticks([42.75,43,44,45,46,47,48,48.75])
axf.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axf.xaxis.set_ticklabels(xxticklabels)
axf.grid(False)

cpw = axw.pcolormesh(P['lon_rho'], P['lat_rho'], P['winter'],vmin=0,vmax=100,cmap=cm.roma_r)
cbaxes = inset_axes(axw, width="5%", height="40%", loc='lower right')
fig.colorbar(cpw, cax=cbaxes, location='left', orientation='vertical',label = 'percent days with \u03A9 ≤ '+str(threshold))

cpsp = axs.pcolormesh(P['lon_rho'], P['lat_rho'], P['spring'],vmin=0,vmax=100,cmap=cm.roma_r)
cbaxes = inset_axes(axs, width="5%", height="40%", loc='lower right')
fig.colorbar(cpsp, cax=cbaxes, location='left', orientation='vertical',label = 'percent days with \u03A9 ≤ '+str(threshold))

cpsu = axsu.pcolormesh(P['lon_rho'], P['lat_rho'], P['summer'],vmin=0,vmax=100,cmap=cm.roma_r)
cbaxes = inset_axes(axsu, width="5%", height="40%", loc='lower right')
fig.colorbar(cpsu, cax=cbaxes, location='left', orientation='vertical',label = 'percent days with \u03A9 ≤ '+str(threshold))

cpf = axf.pcolormesh(P['lon_rho'], P['lat_rho'], P['fall'],vmin=0,vmax=100,cmap=cm.roma_r)
cbaxes = inset_axes(axf, width="5%", height="40%", loc='lower right')
fig.colorbar(cpf, cax=cbaxes, location='left', orientation='vertical',label = 'percent days with \u03A9 ≤ '+str(threshold))

axw.tick_params(axis = "x", labelsize = 14, labelrotation = 0)

axw.set_title('January - March')
axs.set_title('April - June')
axsu.set_title('July - September')
axf.set_title('October - December')

#axf.text(0.82, 0.85,'contours',color='black',weight='normal',fontstyle='italic',transform=axf.transAxes,ha='right')
axf.text(0.82, 0.83,'40 m',color='darkgrey',weight='light',fontstyle='italic',transform=axf.transAxes,ha='right')
axf.text(0.82, 0.81,'80 m',color='grey',weight='light',fontstyle='italic',transform=axf.transAxes,ha='right')
axf.text(0.82, 0.79,'200 m',color='black',weight='light',fontstyle='italic',transform=axf.transAxes,ha='right')
axf.text(0.82, 0.77,'1000 m',color='black',weight='light',fontstyle='italic',transform=axf.transAxes,ha='right')

axsu.yaxis.set_label_position("right")
axsu.yaxis.tick_right()
axsu.set_ylabel(' ')

axs.set_yticklabels([])
axsu.set_yticklabels([])
axf.set_yticklabels([])

## LOWER LIMIT

# add this in as a command line arguments
threshold = 0.5

pn2 = 'seasonal_percent_time_ARAG'+str(threshold)+'_bottom_OA_indicators.pkl'
picklepath2 = fn_o/pn2

with open(picklepath2, 'rb') as fp2:
    P2 = pickle.load(fp2)
    print('loaded file')

# Summer!!!!! 
axsu2 = plt.subplot2grid((3,5), (0,4), colspan=1,rowspan=3)
pfun.add_coast(axsu2)
pfun.dar(axsu2)
axsu2.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
axsu2.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axsu2.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
axsu2.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')

axsu2.set_title('July - September*')
axsu2.set_xlabel('Longitude')
axsu2.set_xlim([xmin, xmax])
axsu2.set_ylim([42.75,48.75])
axsu2.set_xticks(xxticks)
axsu2.set_yticks([42.75,43,44,45,46,47,48,48.75])
axsu2.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axsu2.xaxis.set_ticklabels(xxticklabels)
axsu2.grid(False)   

cpsu2 = axsu2.pcolormesh(P2['lon_rho'], P2['lat_rho'], P2['summer'],vmin=0,vmax=100,cmap=cm.roma_r)
cbaxes = inset_axes(axsu2, width="5%", height="40%", loc='lower right')
fig.colorbar(cpsu2, cax=cbaxes, location='left', orientation='vertical',label = 'percent days with \u03A9 ≤ '+str(threshold))

axsu2.yaxis.set_label_position("right")
axsu2.yaxis.tick_right()
axsu2.set_ylabel('Latitude')

fig.tight_layout()

figname = 'NEW_seasonal_percent_time_ARAG_BOTH.png'
fig.savefig(fn_o / figname)