"""
Plots shelf regions.
"""
# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import sys 
import xarray as xr
from xarray import open_dataset, Dataset
import netCDF4 as nc
from time import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

plot_regions = False
calc_volumes = True 
REGION = 'R1' 

tt0 = time()

# 1 load datasets; assign values
dsm1 = xr.open_dataset('/Users/katehewett/Documents/LO_code/testing_plotting/clip_coords/shelf_mask_15_200m_coastal_clip.nc')
mask_rho = dsm1.mask_rho.values                # 0 = land 1 = water
mask_shelf = dsm1.mask_shelf.values            # 0 = nope 1 = shelf
xrho = dsm1['Lon'].values
yrho = dsm1['Lat'].values
h = dsm1['h'].values
del dsm1 

Ldir = Lfun.Lstart()
fn_i = Ldir['LOo'] / 'extract' / 'cas6_v0_live' 
fn = fn_i / 'corrosive_volume' / 'combined_2017_2022' / 'corrosive_volumes_2017_2022_withgrid.nc'

ds = xr.open_dataset(fn, decode_times=False)     # vols
DA = ds['DA'].values
corrosive_dz = ds['corrosive_dz'].values

ot = ds['ocean_time'].values
days = (ot - ot[0])/86400.
mdays = Lfun.modtime_to_mdate_vec(ot)
mdt = mdates.num2date(mdays) # list of datetimes of data

NT, NR, NC = np.shape(hyp_dz)              # vector of months and years for stats 
#mmonth = np.nan * np.ones((NT))
#myear = np.nan * np.ones((NT))
#for jj in range(NT):
#    mmonth[jj] = mdt[jj].month
#    myear[jj] = mdt[jj].year

#del ds 
#del dsm 

print('Time to load and assign values = %0.2f sec' % (time()-tt0))

tt0 = time()
print('Working on masks ...')
# 2 mask regions
mask_dict = {}
inner_dict = {}
mid_dict = {}
outer_dict = {}

mask_dict[49] = (yrho >=48.75) & (yrho < 49.75) & (mask_shelf == 1) 
mask_dict[48] = (yrho >=47.75) & (yrho < 48.75) & (mask_shelf == 1) 
mask_dict[47] = (yrho >=46.75) & (yrho < 47.75) & (mask_shelf == 1) 
mask_dict[46] = (yrho >=45.75) & (yrho < 46.75) & (mask_shelf == 1) 
mask_dict[45] = (yrho >=44.75) & (yrho < 45.75) & (mask_shelf == 1) 
mask_dict[44] = (yrho >=43.75) & (yrho < 44.75) & (mask_shelf == 1) 
mask_dict[43] = (yrho >=42.75) & (yrho < 43.75) & (mask_shelf == 1) 

inner_dict[49] = (yrho >=48.75) & (yrho < 49.75) & (mask_shelf == 1) & (h >= 15) & (h < 40)
inner_dict[48] = (yrho >=47.75) & (yrho < 48.75) & (mask_shelf == 1) & (h >= 15) & (h < 40)
inner_dict[47] = (yrho >=46.75) & (yrho < 47.75) & (mask_shelf == 1) & (h >= 15) & (h < 40)
inner_dict[46] = (yrho >=45.75) & (yrho < 46.75) & (mask_shelf == 1) & (h >= 15) & (h < 40)
inner_dict[45] = (yrho >=44.75) & (yrho < 45.75) & (mask_shelf == 1) & (h >= 15) & (h < 40)
inner_dict[44] = (yrho >=43.75) & (yrho < 44.75) & (mask_shelf == 1) & (h >= 15) & (h < 40)
inner_dict[43] = (yrho >=42.75) & (yrho < 43.75) & (mask_shelf == 1) & (h >= 15) & (h < 40)

mid_dict[49] = (yrho >=48.75) & (yrho < 49.75) & (mask_shelf == 1) & (h >= 40) & (h < 80)
mid_dict[48] = (yrho >=47.75) & (yrho < 48.75) & (mask_shelf == 1) & (h >= 40) & (h < 80)
mid_dict[47] = (yrho >=46.75) & (yrho < 47.75) & (mask_shelf == 1) & (h >= 40) & (h < 80)
mid_dict[46] = (yrho >=45.75) & (yrho < 46.75) & (mask_shelf == 1) & (h >= 40) & (h < 80)
mid_dict[45] = (yrho >=44.75) & (yrho < 45.75) & (mask_shelf == 1) & (h >= 40) & (h < 80)
mid_dict[44] = (yrho >=43.75) & (yrho < 44.75) & (mask_shelf == 1) & (h >= 40) & (h < 80)
mid_dict[43] = (yrho >=42.75) & (yrho < 43.75) & (mask_shelf == 1) & (h >= 40) & (h < 80)

outer_dict[49] = (yrho >=48.75) & (yrho < 49.75) & (mask_shelf == 1) & (h >= 80) & (h <= 200)
outer_dict[48] = (yrho >=47.75) & (yrho < 48.75) & (mask_shelf == 1) & (h >= 80) & (h <= 200)
outer_dict[47] = (yrho >=46.75) & (yrho < 47.75) & (mask_shelf == 1) & (h >= 80) & (h <= 200)
outer_dict[46] = (yrho >=45.75) & (yrho < 46.75) & (mask_shelf == 1) & (h >= 80) & (h <= 200)
outer_dict[45] = (yrho >=44.75) & (yrho < 45.75) & (mask_shelf == 1) & (h >= 80) & (h <= 200)
outer_dict[44] = (yrho >=43.75) & (yrho < 44.75) & (mask_shelf == 1) & (h >= 80) & (h <= 200)
outer_dict[43] = (yrho >=42.75) & (yrho < 43.75) & (mask_shelf == 1) & (h >= 80) & (h <= 200)

NMASK = len(mask_dict)
print('Time to create mask dict = %0.2f sec' % (time()-tt0))

lat_list = [49, 48, 47, 46, 45, 44, 43]

if calc_volumes == True:
    tt0 = time()
    
    print('Working on volumes ...')
    
    # calculate volumes - no masks yet 
    V = DA * h                  # the total volume of water on rho points
    corrosiveV = DA * corrosive_dz                   
   
    corrVT = np.nan * np.ones((3,NT,NMASK))           # initialize a bunch of vars for saving stuff   
    frac_corrVT = np.nan * np.ones((3,NT,NMASK))
    
    # calc volume for each region and depth 
    Vregion = {}
    Vinner = {}
    Vmid = {}
    Vouter = {}
    
    print('Working on volumes ...')
    
    for mm in lat_list:
        Vregion[mm] = sum(V[mask_dict[mm]])     
        Vinner[mm] = sum(V[inner_dict[mm]]) 
        Vmid[mm] = sum(V[mid_dict[mm]]) 
        Vouter[mm] = sum(V[outer_dict[mm]]) 
        
    for t in range(NT):
        print('Working on file %0.0f ...' % t)
        cvt = corrosiveV[t,:,:].squeeze()
        for mm in range(NMASK):
            corrVT[0,t,mm] = np.nansum(cvt[outer_dict[lat_list[mm]]])
            corrVT[1,t,mm] = np.nansum(cvt[mid_dict[lat_list[mm]]])
            corrVT[2,t,mm] = np.nansum(cvt[inner_dict[lat_list[mm]]])

            frac_corrVT[0,t,mm] = (corrVT[0,t,mm]/Vouter[lat_list[mm]]) * 100
            frac_corrVT[1,t,mm] = (corrVT[1,t,mm]/Vmid[lat_list[mm]]) * 100
            frac_corrVT[2,t,mm] = (corrVT[2,t,mm]/Vinner[lat_list[mm]]) * 100

    print('Time to calc volumes = %0.2f sec' % (time()-tt0)) # 300 seconds 
    
    ds1 = Dataset()
    ds1['corrosive_volume'] = (('shelf','time','region'), mildVT, {'units':'cubic meter', 'long_name': 'corrosive volume leq1; shelf dimension: outer 0; mid 1; inner 2'})
    
    ds1['frac_corrosive'] = (('shelf','time','region'), frac_mildV, {'units':'cubic meter', 'long_name': 'fraction of shelf/region: corrosive volume leq1; shelf dimension: outer 0; mid 1; inner 2'})
    
    ds1.to_netcdf('/Users/katehewett/Documents/LKH_output/tests/percent_corrosive/2017_2022_corrosiveVol_percentages.nc')
    
if plot_regions:
    tt0 = time()
    
    Rcolors = ['#73210D','#9C6B2F','#C5B563','#A9A9A9','#77BED0','#4078B3','#112F92'] #roma w grey not green

    plt.close('all')
    fs=16
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(18,10))

    # map
    ax = fig.add_subplot(131)
    pfun.add_coast(ax)
    pfun.dar(ax)
    ax.axis([-128, -123, 42.75, 49.75])
    
    ax.contour(xrho,yrho,h, [40],
    colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
    ax.contour(xrho,yrho,h, [80],
    colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
    ax.contour(xrho,yrho,h, [130],
    colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
    ax.contour(xrho,yrho,h, [2000],
    colors=['black'], linewidths=1, linestyles='solid')
    ax.text(.33,.19,'40 m',color='lightgrey',weight='bold',transform=ax.transAxes,ha='right')
    ax.text(.33,.16,'80 m',color='grey',weight='bold',transform=ax.transAxes,ha='right')
    ax.text(.33,.13,'130 m',color='dimgrey',weight='bold',transform=ax.transAxes,ha='right')
    ax.text(.33,.1,'*200 m',color='black',weight='normal',fontstyle = 'italic',transform=ax.transAxes,ha='right')
    ax.text(.33,.07,'2000 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
    ax.set_title('Area of Calculation')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks([-128, -127, -126, -125, -124, -123])
    #ax.grid(True)
    ax.set_xticklabels(['-128',' ','-126',' ','-124',' '])
      
    ii = 0
    for mm in lat_list:
        pts2 = ax.scatter(xrho[mask_dict[mm]], yrho[mask_dict[mm]], s=2, c = Rcolors[ii])
        ii = ii+1
    
    print('Time to create regional plot = %0.2f sec' % (time()-tt0))
    
    fig.savefig('/Users/katehewett/Documents/LKH_output/tests/regions.png', dpi=720)
    








