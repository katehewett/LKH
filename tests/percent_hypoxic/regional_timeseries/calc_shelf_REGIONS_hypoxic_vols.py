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
fn = fn_i / 'hypoxic_volume' / 'combined_2017_2022' / 'hypoxic_volumes_2017_2022_withgrid.nc'

ds = xr.open_dataset(fn, decode_times=False)     # vols
DA = ds['DA'].values
mild_dz = ds['mild_dz'].values
hyp_dz = ds['hyp_dz'].values
severe_dz = ds['severe_dz'].values
anoxic_dz = ds['anoxic_dz'].values

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
    mildV = DA * mild_dz        # the hyp volume on each rho point
    hypV = DA * hyp_dz              
    severeV = DA * severe_dz        
    anoxicV = DA * anoxic_dz        
   
    hypVT = np.nan * np.ones((3,NT,NMASK))           # initialize a bunch of vars for saving stuff 
    #frac_hypV = np.nan * np.ones((3,NT,NMASK))
    
    severeVT = np.nan * np.ones((3,NT,NMASK))
    #frac_severeV = np.nan * np.ones((3,NT,NMASK))

    mildVT = np.nan * np.ones((3,NT,NMASK))
    #frac_mildV = np.nan * np.ones((3,NT,NMASK))
    
    anoxicVT = np.nan * np.ones((3,NT,NMASK))
    #frac_anoxicV = np.nan * np.ones((3,NT,NMASK))
    
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
        mvt = mildV[t,:,:].squeeze()
        hvt = hypV[t,:,:].squeeze()
        svt = severeV[t,:,:].squeeze()
        avt = anoxicV[t,:,:].squeeze()
        for mm in range(NMASK):
            mildVT[0,t,mm] = np.nansum(mvt[outer_dict[lat_list[mm]]])
            mildVT[1,t,mm] = np.nansum(mvt[mid_dict[lat_list[mm]]])
            mildVT[2,t,mm] = np.nansum(mvt[inner_dict[lat_list[mm]]])
            
            hypVT[0,t,mm] = np.nansum(hvt[outer_dict[lat_list[mm]]])
            hypVT[1,t,mm] = np.nansum(hvt[mid_dict[lat_list[mm]]])
            hypVT[2,t,mm] = np.nansum(hvt[inner_dict[lat_list[mm]]])
            
            severeVT[0,t,mm] = np.nansum(svt[outer_dict[lat_list[mm]]])
            severeVT[1,t,mm] = np.nansum(svt[mid_dict[lat_list[mm]]])
            severeVT[2,t,mm] = np.nansum(svt[inner_dict[lat_list[mm]]])
            
            anoxicVT[0,t,mm] = np.nansum(avt[outer_dict[lat_list[mm]]])
            anoxicVT[1,t,mm] = np.nansum(avt[mid_dict[lat_list[mm]]])
            anoxicVT[2,t,mm] = np.nansum(avt[inner_dict[lat_list[mm]]])
            
            frac_mildV[0,t,mm] = (mildVT[0,t,mm]/Vouter[lat_list[mm]]) * 100
            frac_mildV[1,t,mm] = (mildVT[1,t,mm]/Vmid[lat_list[mm]]) * 100
            frac_mildV[2,t,mm] = (mildVT[2,t,mm]/Vinner[lat_list[mm]]) * 100
            
            frac_hypV[0,t,mm] = (hypVT[0,t,mm]/Vouter[lat_list[mm]]) * 100
            frac_hypV[1,t,mm] = (hypVT[1,t,mm]/Vmid[lat_list[mm]]) * 100
            frac_hypV[2,t,mm] = (hypVT[2,t,mm]/Vinner[lat_list[mm]]) * 100
            
            frac_severeV[0,t,mm] = (severeVT[0,t,mm]/Vouter[lat_list[mm]]) * 100
            frac_severeV[1,t,mm] = (severeVT[1,t,mm]/Vmid[lat_list[mm]]) * 100
            frac_severeV[2,t,mm] = (severeVT[2,t,mm]/Vinner[lat_list[mm]]) * 100
            
            frac_anoxicV[0,t,mm] = (anoxicVT[0,t,mm]/Vouter[lat_list[mm]]) * 100
            frac_anoxicV[1,t,mm] = (anoxicVT[1,t,mm]/Vmid[lat_list[mm]]) * 100
            frac_anoxicV[2,t,mm] = (anoxicVT[2,t,mm]/Vinner[lat_list[mm]]) * 100

    print('Time to calc volumes = %0.2f sec' % (time()-tt0)) # 300 seconds 
    
    ds1 = Dataset()
    ds1['mild_volume'] = (('shelf','time','region'), mildVT, {'units':'cubic meter', 'long_name': 'mild hypoxic volume; shelf dimension: outer 0; mid 1; inner 2'})
    ds1['intermediate_volume'] = (('shelf','time','region'), hypVT, {'units':'cubic meter', 'long_name': 'intermediate hypoxic volume; shelf dimension: outer 0; mid 1; inner 2'})
    ds1['severe_volume'] = (('shelf','time','region'), severeVT, {'units':'cubic meter', 'long_name': 'severe hypoxic volume; shelf dimension: outer 0; mid 1; inner 2'})
    ds1['anoxic_volume'] = (('shelf','time','region'), anoxicVT, {'units':'cubic meter', 'long_name': 'anoxic hypoxic volume; shelf dimension: outer 0; mid 1; inner 2'})
    
    ds1['frac_mildV'] = (('shelf','time','region'), frac_mildV, {'units':'cubic meter', 'long_name': 'fraction of shelf/region: mild hypoxic volume; shelf dimension: outer 0; mid 1; inner 2'})
    ds1['frac_hypV'] = (('shelf','time','region'), frac_hypV, {'units':'cubic meter', 'long_name': 'fraction of shelf/region: intermediate hypoxic volume; shelf dimension: outer 0; mid 1; inner 2'})
    ds1['frac_severeV'] = (('shelf','time','region'), frac_severeV, {'units':'cubic meter', 'long_name': 'fraction of shelf/region: severe hypoxic volume; shelf dimension: outer 0; mid 1; inner 2'})
    ds1['frac_anoxicV'] = (('shelf','time','region'), frac_anoxicV, {'units':'cubic meter', 'long_name': 'fraction of shelf/region: anoxic hypoxic volume; shelf dimension: outer 0; mid 1; inner 2'})

    ds1.to_netcdf('/Users/katehewett/Documents/LKH_output/tests/percent_hypoxic/2017_2022_hypoxicVol_percentages_2.nc')
    
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
    








