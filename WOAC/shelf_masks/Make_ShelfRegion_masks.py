"""
This is a one-off script used to produce a mask 
for the inner shelf w/in the OA_indicators domain

!!! This won't work as-is on apogee b/c not using Ldir
Setup matches but paths are /user/kmhewett not /dat1/etc...

This documents the masking process for one plot :: 
General flow:
- From the clipped shelf_domain OA_indicators_job 
- make a flag/mask for 15 - 40m (flag rho_points in that range)
- Then manually exclude a few points off of... (1) VI and (2) central OR

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
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

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
#axw.axis([-124.5, -124, 44, 45])
axw.plot(dse['lon_rho'],dse['lat_rho'],color='pink',marker='.', linestyle='none')
axw.plot(xrho[mask_shelf==1],yrho[mask_shelf==1],color='blue',marker='.', linestyle='none')
axw.plot(smolx[smask_shelf==1],smoly[smask_shelf==1],color='green',marker='.', linestyle='none')

pfun.add_coast(axw)
pfun.dar(axw)
axw.set_xticks([-128, -127, -126, -125, -124, -123])

axw.set_title('Area of Calculation')
axw.set_ylabel('Latitude')
axw.set_xlabel('Longitude')
axw.set_xticklabels(['-128',' ','-126',' ','-124',' '])
axw.tick_params(axis = "x", labelsize = 8, labelrotation = 0)

# make a mask for the zone between h = 15:40m
mask_inner = np.ones(np.shape(smask_shelf))
mask_inner[smask_shelf==0] = 0 
mask_inner[mask_rho2==0] = 0
mask_inner[h2>40] = 0
mask_inner[h2<15] = 0

# make a mask for the zone between h = 40:80m
mask_mid = np.ones(np.shape(smask_shelf))
mask_mid[smask_shelf==0] = 0 
mask_mid[mask_rho2==0] = 0
mask_mid[(h2>80) | (h2==80)] = 0
mask_mid[(h2<40) | (h2==40)] = 0

# make a mask for the zone between h = 80 - 130m
mask_outer = np.ones(np.shape(smask_shelf))
mask_outer[smask_shelf==0] = 0 
mask_outer[mask_rho2==0] = 0
mask_outer[h2>200] = 0
mask_outer[h2<80] = 0


# remove VI points 
mask_inner[smoly[:,0]>48.450,:] = 0
mask_mid[smoly[:,0]>48.450,:] = 0
mask_outer[smoly[:,0]>48.450,:] = 0

# remove central oregon points off the "main" 40m shelf isobath 
ix0 = zfun.find_nearest_ind(smolx[0,:],-124.5)
ix1 = zfun.find_nearest_ind(smolx[0,:],-124.3)

iy0 = zfun.find_nearest_ind(smoly[:,0],44.4)
iy1 = zfun.find_nearest_ind(smoly[:,0],44.6) 

mask_inner[iy0:iy1+1,ix0:ix1+1] = 0


axw.plot(smolx[mask_inner==1],smoly[mask_inner==1],color='yellow',marker='.', linestyle='none')
axw.plot(smolx[mask_mid==1],smoly[mask_mid==1],color='cyan',marker='.', linestyle='none')
axw.plot(smolx[mask_outer==1],smoly[mask_outer==1],color='white',marker='.', linestyle='none')
#axw.contour(xrho,yrho,h, [30, 50],colors=['red'], linewidths=1, linestyles='solid',alpha=0.4)



# remove VI on north side of JdF canyon
"""
==============
Lasso Selector
==============

Interactively selecting data points with the lasso tool.

This examples plots a scatter plot. You can then select a few points by drawing
a lasso loop around the points on the graph. To draw, just click
on the graph, hold, and drag it around the points you need to select.
"""

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    subplot_kw = dict(xlim=(-125.75,-124.5), ylim=(47.9, 48.5), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(smolx[mask_outer==1],smoly[mask_outer==1], s=80)
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

a = selector.xys[selector.ind].data

sys.exit()
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.LassoSelector`
#    - `matplotlib.path.Path`




sys.exit()
# save the dataset 
dsm2 = Dataset()

dsm2['lat_rho'] = (('eta_rho', 'xi_rho'),smoly,{'units':'degree_north'})
dsm2['lat_rho'].attrs['standard_name'] = 'grid_latitude_at_cell_center'
dsm2['lat_rho'].attrs['long_name'] = 'latitude of RHO-points'
dsm2['lat_rho'].attrs['field'] = 'lat_rho'

dsm2['lon_rho'] = (('eta_rho', 'xi_rho'),smolx,{'units': 'degree_east'})
dsm2['lon_rho'].attrs['standard_name'] = 'grid_longitude_at_cell_center'
dsm2['lon_rho'].attrs['long_name'] = 'longitude of RHO-points'
dsm2['lon_rho'].attrs['field'] = 'lon_rho'
 
dsm2['h'] = (('eta_rho', 'xi_rho'),h2,{'units': 'm'})
dsm2['h'].attrs['standard_name'] = 'sea_floor_depth'
dsm2['h'].attrs['long_name'] = 'time_independent bathymetry'
dsm2['h'].attrs['field'] = 'bathymetry'
dsm2['h'].attrs['grid'] =  'cas7_t0_x4b'

dsm2['mask_h40m'] = (('eta_rho', 'xi_rho'),mask_h40m,{'units': 'm'})
dsm2['mask_h40m'].attrs['standard_name'] = 'h40m_mask_at_cell_center'
dsm2['mask_h40m'].attrs['long_name'] = 'mask on RHO-points'
dsm2['mask_h40m'].attrs['flag_values'] = np.array([0.,1.])
dsm2['mask_h40m'].attrs['flag_meanings'] = 'outside 50m<h<30m'
dsm2['mask_h40m'].attrs['grid'] =  'cas7_t0_x4b'

fn_o =  '/Users/katehewett/Documents/LKH_data/shelf_masks/OA_indicators/OA_indicators_h40m_mask.nc'
dsm2.to_netcdf(fn_o)

sys.exit()

## smaller OA_indicator domain
#hm40 = abs(h2-40)
## make the off-shelf (and land) values huge
#hm40[smask_shelf==0]=9999
#hm40[mask_rho2==0]=9999

#idx = np.argmin(hm40,axis=1,keepdims=True)
#xdx = np.take_along_axis(smolx,idx,axis=1)
#ydx = np.take_along_axis(smoly,idx,axis=1)

