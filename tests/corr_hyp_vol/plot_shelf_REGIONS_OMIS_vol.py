"""
Plots shelf region hypoxia 
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

plot_regions = True 

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

# plot results 
if (plot_regions == True):     
    tt0 = time()
    ax1 = plt.subplot2grid((3,1), (0,0), colspan=3)
    ax2 = plt.subplot2grid((3,1), (1,0), colspan=2)
    ax3 = plt.subplot2grid((3,1), (2,0), colspan=2)
    
    ZZ = hypVT[:,3]*0
    ax1.plot(mdt,frac_hypV[0,:,0].squeeze(), color='tab:red', alpha=0.6)
    ax2.fill_between(mdt,ZZ,hypVT[:,3]/1e9, color=Rcolors[3], alpha=0.6)
    
    ax1.plot(mdt, hypVT[:,0]/1e9, color=Rcolors[0], linewidth=2, alpha=0.8)
    ax1.plot(mdt, hypVT[:,1]/1e9, color=Rcolors[1], linewidth=2, alpha=0.8)
    ax1.plot(mdt, hypVT[:,2]/1e9, color=Rcolors[2], linewidth=2, alpha=0.8)
    
    ax2.plot(mdt, hypVT[:,4]/1e9, color=Rcolors[4], linewidth=2, alpha=0.8)
    ax2.plot(mdt, hypVT[:,5]/1e9, color=Rcolors[5], linewidth=2, alpha=0.8)
    ax2.plot(mdt, hypVT[:,6]/1e9, color=Rcolors[6], linewidth=2, alpha=0.8)
     
    #ax1.fill_between(mdt,ZZ,frac_severeV[:,3], color=Rcolors[3], alpha=0.6)
    #ax2.fill_between(mdt,ZZ,frac_severeV[:,3], color=Rcolors[3], alpha=0.6)
    
    #ax1.plot(mdt, frac_severeV[:,0], color=Rcolors[0], linewidth=2, alpha=0.8)
    #ax1.plot(mdt, frac_severeV[:,1], color=Rcolors[1], linewidth=2, alpha=0.8)
    #ax1.plot(mdt, frac_severeV[:,2], color=Rcolors[2], linewidth=2, alpha=0.8)
    
    #ax2.plot(mdt, frac_severeV[:,4], color=Rcolors[4], linewidth=2, alpha=0.8)
    #ax2.plot(mdt, frac_severeV[:,5], color=Rcolors[5], linewidth=2, alpha=0.8)
    #ax2.plot(mdt, frac_severeV[:,6], color=Rcolors[6], linewidth=2, alpha=0.8)
          
    ax1.set_title('Shelf water anoxia')
    #ax2.set_title('percent of total shelf')
    
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.set_ylabel('volume $km^3$')
    #ax1.set_ylabel('percent total shelf')
    
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel('volume $km^3$')
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

    ax1.set_ylim(0,1)
    ax2.set_ylim(0,1)

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


    
    #fig.savefig('/Users/katehewett/Documents/LO_figures/temp_from_LO_code/regions_anoxia_NEW.png')
    
    #fig.tight_layout()
    #plt.show()






