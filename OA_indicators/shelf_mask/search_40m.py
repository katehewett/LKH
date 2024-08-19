"""
This is a one-off script used to produce a plot
requested from a June2024 meeting:: 

The goal is to grab the xy coords for the ~40m isobath 
to then use in an extraction for bottom and surface data 
at those points --> to then calc corrosive time durations

It's for creating a specific plot for the OA_indicators project
lat/lons listed under /LO_user/extract/corrosive_volume/job_list.py

This code is ugly, becarefffullll :) It's not friendly  
It just documents the masking process for one plot :: General flow:
- From the clipped shelf_domain OA_indicators_job 
- make a flag/mask for 40m contour using argmin and find_along axis
- Then use a very messy lasso function to exclude a few points off of:
--- JdF canyon/VI points + a few points off oregon that arg min grabs

- this ends in saving a rough lat/lon for the ~40m isobath for the OA_indicator job
"""

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys 
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np
from xarray import Dataset

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cmcrameri import cm as cm2
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

tt0 = time()
 
# 1 load datasets; assign values
dsm1 = xr.open_dataset('/Users/katehewett/Documents/LKH_data/shelf_masks/LO_domain/shelf_mask_15_200m_coastal_clip.nc')
mask_rho = dsm1.mask_rho.values                # 0 = land 1 = water
mask_shelf = dsm1.mask_shelf.values            # 0 = nope 1 = shelf
xrho = dsm1['Lon'].values
yrho = dsm1['Lat'].values
h = dsm1['h'].values
#del dsm1 

# open one file to get the lat lon of the extraction for 
# the OA_indicators job:
aa = [-125.5, -123.5, 42.75, 48.75]
dse = xr.open_dataset('/Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/corrosive_volume/OA_indicators_2013.01.01_2013.12.31/OA_indicators_corrosive_volume_2013.01.01_2013.12.31.nc')

Y = yrho[:,0]
X = xrho[0,:]

ilon0 = zfun.find_nearest_ind(X,dse['lon_rho'].values[0,0])
ilon1 = zfun.find_nearest_ind(X,dse['lon_rho'].values[0,-1])

ilat0 = zfun.find_nearest_ind(Y,dse['lat_rho'].values[0,0])
ilat1 = zfun.find_nearest_ind(Y,dse['lat_rho'].values[-1,0]) 

smolx = xrho[ilat0:ilat1+1,ilon0:ilon1+1]
smoly = yrho[ilat0:ilat1+1,ilon0:ilon1+1]
smask_shelf = mask_shelf[ilat0:ilat1+1,ilon0:ilon1+1]
h2 = h[ilat0:ilat1+1,ilon0:ilon1+1]
mask_rho2 = mask_rho[ilat0:ilat1+1,ilon0:ilon1+1]

#Rcolors = ['#73210D','#9C6B2F','#C5B563','#A9A9A9','#77BED0','#4078B3','#112F92'] #roma w grey not green #idx 3
    
plt.close('all')
fs=16
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

axw = plt.subplot2grid((3,4), (0,0), colspan=1,rowspan=3)
pfun.add_coast(axw)
pfun.dar(axw)
axw.axis([-128, -123, 42.75, 49.75])
#axw.plot(dse['lon_rho'],dse['lat_rho'],color='pink',marker='.', linestyle='none')
#axw.plot(xrho[mask_shelf==1],yrho[mask_shelf==1],color='blue',marker='.', linestyle='none')
#axw.plot(smolx[smask_shelf==1],smoly[smask_shelf==1],color='green',marker='.', linestyle='none')
axw.contour(xrho,yrho,h, [40],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axw.set_xticks([-128, -127, -126, -125, -124, -123])

axw.set_title('Area of Calculation')
axw.set_ylabel('Latitude')
axw.set_xlabel('Longitude')
axw.set_xticklabels(['-128',' ','-126',' ','-124',' '])
axw.tick_params(axis = "x", labelsize = 14, labelrotation = 0)

# smaller OA_indicator domain
hm40 = abs(h2-40)
# make the off-shelf (and land) values huge
hm40[smask_shelf==0]=9999
hm40[mask_rho2==0]=9999

idx = np.argmin(hm40,axis=1,keepdims=True)
xdx = np.take_along_axis(smolx,idx,axis=1)
ydx = np.take_along_axis(smoly,idx,axis=1)

# initialize a mask for the points nearest to the ~40m isobath
mask_h40 = np.ones(np.shape(smask_shelf))*0

# assign 1's to the xrhos we flagged with idx above
mask_h40[:,idx] = 1

axw.plot(xdx,ydx,color='red',marker='.',markersize = 0.2, linestyle='none')


plt.close('all')
## messy way to lasso the off points in mask

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


# lasso off points 
class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

#[-128, -123, 42.75, 49.75]
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    data = np.stack((xdx,ydx),axis=1).squeeze()

    subplot_kw = dict(xlim=(-128, -123), ylim=(42.75, 49.75), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], s=2)
    selector = SelectFromCollection(ax, pts)

    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            print(selector.xys[selector.ind])
            selector.disconnect()
            ax.set_title("")
            fig.canvas.draw()
            
    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")

    plt.show()


xX = selector.xys[selector.ind].data[:,0].squeeze()
xY = selector.xys[selector.ind].data[:,1].squeeze()

plt.close('all')
fs=16
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

axw = plt.subplot2grid((3,4), (0,0), colspan=1,rowspan=3)
pfun.add_coast(axw)
pfun.dar(axw)
axw.axis([-128, -123, 42.75, 49.75])
#axw.plot(dse['lon_rho'],dse['lat_rho'],color='pink',marker='.', linestyle='none')
#axw.plot(xrho[mask_shelf==1],yrho[mask_shelf==1],color='blue',marker='.', linestyle='none')
#axw.plot(smolx[smask_shelf==1],smoly[smask_shelf==1],color='green',marker='.', linestyle='none')
axw.contour(xrho,yrho,h, [40],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axw.set_xticks([-128, -127, -126, -125, -124, -123])

axw.set_title('Area of Calculation')
axw.set_ylabel('Latitude')
axw.set_xlabel('Longitude')
axw.set_xticklabels(['-128',' ','-126',' ','-124',' '])
axw.tick_params(axis = "x", labelsize = 14, labelrotation = 0)

axw.plot(xdx,ydx,color='blue',marker='.',markersize = 0.2, linestyle='none')
axw.plot(xX,xY,color='red',marker='.',markersize = 0.2, linestyle='none')



