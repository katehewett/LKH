"""
Plots shelf regions.
"""
# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import sys 
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

# select what to do 
plot_shelf = False
plot_regions = True
calc_volumes = True

tt0 = time()
# 1 load datasets; assign values
Ldir = Lfun.Lstart()

fn_i = Ldir['LOo'] / 'extract' / 'cas6_v0_live' 
fn = fn_i / 'corrosive_volume' / 'combined_2017_2022' / 'corrosive_volumes_2017_2022_withgrid.nc'

ds = xr.open_dataset(fn, decode_times=False)     # vols
#ds2 = xr.open_dataset(fn2)                      # grid info and zeta
#dsxp = xr.open_dataset('/Users/katehewett/Documents/LO_roms/cas6_v0_live/f2022.08.08/ocean_his_0003.nc')
dsm = xr.open_dataset('/Users/katehewett/Documents/LO_code/testing_plotting/clip_coords/shelf_mask_15_200m_VIclip.nc')
# shelf mask

#xrho = ds2['lon_rho'].values
#yrho = ds2['lat_rho'].values
xrho = ds['Lon'].values
yrho = ds['Lat'].values
h = ds['h'].values
#h1 = np.ones_like(h)
DA = ds['DA'].values
#dzr = ds['dzr'].values
#zeta = ds['zeta'].values 
mask_rho = ds['mask_rho'].values
corrosive_dz = ds['corrosive_dz'].values

mask_shelf = dsm['mask_shelf'].values

ot = ds['ocean_time'].values
days = (ot - ot[0])/86400.
mdays = Lfun.modtime_to_mdate_vec(ot)
mdt = mdates.num2date(mdays) # list of datetimes of data

NT, NR, NC = np.shape(corrosive_dz)              # vector of months and years for stats 
mmonth = np.nan * np.ones((NT))
myear = np.nan * np.ones((NT))
for jj in range(NT):
    mmonth[jj] = mdt[jj].month
    myear[jj] = mdt[jj].year

del ds 
del dsm 

print('Time to load and assign values = %0.2f sec' % (time()-tt0))

tt0 = time()
# 2 mask regions
mask_dict = {}
mask_dict[49] = (yrho >=48.75) & (yrho < 49.75) & (mask_shelf == 1)
mask_dict[48] = (yrho >=47.75) & (yrho < 48.75) & (mask_shelf == 1)
mask_dict[47] = (yrho >=46.75) & (yrho < 47.75) & (mask_shelf == 1)
mask_dict[46] = (yrho >=45.75) & (yrho < 46.75) & (mask_shelf == 1)
mask_dict[45] = (yrho >=44.75) & (yrho < 45.75) & (mask_shelf == 1)
mask_dict[44] = (yrho >=43.75) & (yrho < 44.75) & (mask_shelf == 1)
mask_dict[43] = (yrho >42.75) & (yrho < 43.75) & (mask_shelf == 1)
NMASK = len(mask_dict)
print('Time to create mask dict = %0.2f sec' % (time()-tt0))

lat_list = [49, 48, 47, 46, 45, 44, 43]

if plot_regions:
    tt0 = time()
    #Rcolors = ['#DCD8FB', '#7388C6', '#2C3A61', '#0E1613', '#2E4C2C', '#6D9D60', '#DDE5A3']
    #Rcolors = ['#73210D','#9C6B2F','#C5B563','#C7E9C6','#77BED0','#4078B3','#112F92'] #roma
    Rcolors = ['#73210D','#9C6B2F','#C5B563','#A9A9A9','#77BED0','#4078B3','#112F92'] #roma w grey not green #idx 3
    
    # PLOTTING
    #cmap = cmocean.cm.haline_r
    #cmap = cm.jet_r

    plt.close('all')
    fs=16
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(18,10))

    # map
    ax = fig.add_subplot(131)
    pfun.add_coast(ax)
    pfun.dar(ax)
    ax.axis([-130, -122, 42, 52])
    
    ax.contour(xrho,yrho,h, [100, 200, 2000],
    colors=['grey','dimgrey','black'], linewidths=1, linestyles='solid')
    ax.text(.33,.16,'100 m',color='grey',weight='bold',transform=ax.transAxes,ha='right')
    ax.text(.33,.13,'200 m',color='dimgrey',weight='bold',transform=ax.transAxes,ha='right')
    ax.text(.33,.1,'2000 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
    ax.set_title('Area of Calculation')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks([-130, -128, -126, -124, -122])
    ax.grid(True)

    #pts2 = ax.scatter(xrho[mask_dict[1]], yrho[mask_dict[1]], s=2, c = '#8D9FC5')
    
    ii = 0
    for mm in lat_list:
        pts2 = ax.scatter(xrho[mask_dict[mm]], yrho[mask_dict[mm]], s=2, c = Rcolors[ii])
        ii = ii+1
    
    #fig.tight_layout()
    #fig.savefig('/Users/katehewett/Documents/LO_figures/temp_from_LO_code/shelf_regions_NEW.png')
    
    print('Time to create regional plot = %0.2f sec' % (time()-tt0))
    

if (calc_volumes == True & plot_regions == True):
    tt0 = time()
    
    # calculate volumes - no masks yet 
    #V = DA * dzrm            # total V water on rho points (after pmac mtg -make it simple- just use h)
    V = DA * h                # the total volume of water on rho points
    corrV = DA * corrosive_dz        # the hyp volume on each rho point   
    
    #hypVm = np.nan * np.ones((NT, NR, NC))    # initialize a bunch of vars for saving stuff 
    corrVT = np.nan * np.ones((NT,NMASK))
    #severeVm = np.nan * np.ones((NT, NR, NC))
    frac_corrV = np.nan * np.ones((NT,NMASK))
    
    # whole shelf, mask where off shelf 
    V = np.ma.masked_where((yrho < 42.75) | (yrho > 49.75) | (mask_shelf != 1), V) 
    V = zfun.fillit(V)
    VT = np.nansum(V)      # total shelf volume
    for t in range(NT):
        print('Working on file %0.0f ...' % t)
        cvt = corrV[t,:,:].squeeze()
        for mm in range(NMASK):
            corrVT[t,mm] = np.nansum(cvt[mask_dict[lat_list[mm]]])
            frac_corrV[t,mm] = (corrVT[t,mm]/VT) * 100
        
    print('Time to calc volumes = %0.2f sec' % (time()-tt0))

# plot results 
if (calc_volumes == True) & (plot_regions == True):     
    tt0 = time()
    ax1 = plt.subplot2grid((2,3), (0,1), colspan=2)
    ax2 = plt.subplot2grid((2,3), (1,1), colspan=2)
    
    #for mm in range(NMASK-1):
    #    ax1.plot(mdt, hypVT[:,mm]/10e9, color=Rcolors[mm], linewidth=2)  
    
    #    ax2.plot(mdt, frac_hypV[:,mm], color=Rcolors[mm], linewidth=2)
    #    #pts2 = ax1.scatter(mdt, hypVT[:,mm]/10e9, s=2, c = Rcolors[mm])
    #    #pts2 = ax2.scatter(mdt, frac_hypV[:,mm], s=2, c = Rcolors[mm])
    
    ZZ = corrVT[:,3]*0
    ax1.fill_between(mdt,ZZ,corrVT[:,3]/10e9, color=Rcolors[3], alpha=0.6)
    ax2.fill_between(mdt,ZZ,corrVT[:,3]/10e9, color=Rcolors[3], alpha=0.6)
    
    ax1.plot(mdt, corrVT[:,0]/10e9, color=Rcolors[0], linewidth=2, alpha=0.8)
    ax1.plot(mdt, corrVT[:,1]/10e9, color=Rcolors[1], linewidth=2, alpha=0.8)
    ax1.plot(mdt, corrVT[:,2]/10e9, color=Rcolors[2], linewidth=2, alpha=0.8)
    
    ax2.plot(mdt, corrVT[:,4]/10e9, color=Rcolors[4], linewidth=2, alpha=0.8)
    ax2.plot(mdt, corrVT[:,5]/10e9, color=Rcolors[5], linewidth=2, alpha=0.8)
    ax2.plot(mdt, corrVT[:,6]/10e9, color=Rcolors[6], linewidth=2, alpha=0.8)
     
    #ax1.fill_between(mdt,ZZ,frac_corrV[:,3], color=Rcolors[3], alpha=0.6)
    #ax2.fill_between(mdt,ZZ,frac_corrV[:,3], color=Rcolors[3], alpha=0.6)
    
    #ax1.plot(mdt, frac_corrV[:,0], color=Rcolors[0], linewidth=2, alpha=0.8)
    #ax1.plot(mdt, frac_corrV[:,1], color=Rcolors[1], linewidth=2, alpha=0.8)
    #ax1.plot(mdt, frac_corrV[:,2], color=Rcolors[2], linewidth=2, alpha=0.8)
    
    #ax2.plot(mdt, frac_corrV[:,4], color=Rcolors[4], linewidth=2, alpha=0.8)
    #ax2.plot(mdt, frac_corrV[:,5], color=Rcolors[5], linewidth=2, alpha=0.8)
    #ax2.plot(mdt, frac_corrV[:,6], color=Rcolors[6], linewidth=2, alpha=0.8)
          
    ax1.set_title('Shelf water corrosive volume, \u03A9 ag $\leq $ 1')
    #ax2.set_title('percent of total shelf')
    
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.set_ylabel('volume x10 $km^3$')
    #ax1.set_ylabel('percent total shelf')
    
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel('volume x10 $km^3$')
    #ax2.set_ylabel('percent total shelf')
    
    #ax1.legend(['volume', 'percentage'])
    
    ax1.grid(True)
    ax2.grid(True)
    
    ax1.set_xticks([datetime(2017,1,1),datetime(2017,7,1),datetime(2018,1,1), datetime(2018,7,1),
    datetime(2019,1,1),datetime(2019,7,1), datetime(2020,1,1),datetime(2020,7,1),
    datetime(2021,1,1),datetime(2021,7,1),datetime(2022,1,1),datetime(2022,7,1),datetime(2022,12,31)])
    
    ax1.set_xticklabels([' ','Jul17',' ','Jul18',
    ' ','Jul19',' ','Jul20',
    ' ','Jul21',' ','Jul22',' '])
    ax1.tick_params(axis = "x", labelsize = 14, labelrotation = 0)

    ax1.set_xlim(mdt[0], mdt[-1])
    ax2.set_xlim(mdt[0], mdt[-1])

    ax1.set_ylim(0,80)
    ax2.set_ylim(0,80)

    ax2.set_xticks([datetime(2017,1,1),datetime(2017,7,1),datetime(2018,1,1), datetime(2018,7,1),
    datetime(2019,1,1),datetime(2019,7,1), datetime(2020,1,1),datetime(2020,7,1),
    datetime(2021,1,1),datetime(2021,7,1),datetime(2022,1,1),datetime(2022,7,1),datetime(2022,12,31)],)
    
    ax2.set_xticklabels(['Jan17',' ','Jan18',' ',
    'Jan19',' ','Jan20',' ',
    'Jan21',' ','Jan22',' ',' '])
    
    ax2.tick_params(axis = "x", labelsize = 14, labelrotation = 0)
    
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    
    print('Time to plot volumes = %0.2f sec' % (time()-tt0))
    
    fig.savefig('/Users/katehewett/Documents/LO_figures/temp_from_LO_code/regions_corrosive_NEW.png')
    
    #fig.tight_layout()
    #plt.show()






