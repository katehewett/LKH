'''

profile plots 1/month 

'''

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
from datetime import timedelta
import pickle 

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from lo_tools import Lfun

Ldir = Lfun.Lstart()

# TODO put to args 
thisyr = 2017 

vel = 'U'
vel = 'V'

moor = 'CE09OSSM' 

in_dir = Ldir['parent'] / 'LKH_output' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'monthly_stats'
fn_1 = moor + '_' + vel + 'monthly_surfacebuoy_VELPTA_'+str(thisyr)+'.pkl'
fn_7 = moor + '_' + vel + 'monthly_nsif_VELPTA_'+str(thisyr)+'.pkl'
fn_mfd = moor + '_' + vel + 'monthly_mfd_ADCP_'+str(thisyr)+'.pkl'
fn_nsif = moor + '_' + vel + 'monthly_nsif_ADCP_'+str(thisyr)+'.pkl'
model2 = moor + '_' + vel + 'monthly_'+str(thisyr)+'_cas7_t1_x4a.pkl'
model1 = moor + '_' + vel + 'monthly_'+str(thisyr)+'_cas7_t0_x4b.pkl'

out_dir = Ldir['parent'] / 'LKH_output' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'monthly_stats' / 'plots'
if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)

plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(14,8))
fig.set_size_inches(14,8, forward=False)

ax1 = plt.subplot2grid((5,1), (0,0), colspan=1,rowspan=1)
ax2 = plt.subplot2grid((5,1), (1,0), colspan=1,rowspan=1)
ax3 = plt.subplot2grid((5,1), (2,0), colspan=1,rowspan=1)
ax4 = plt.subplot2grid((5,1), (3,0), colspan=1,rowspan=1)
ax5 = plt.subplot2grid((5,1), (4,0), colspan=1,rowspan=1)

fig.tight_layout()


sys.exit()


axes_dict = {
    'ax1': ax1,
    'ax2': ax2,
    'ax3': ax3,
    'ax4': ax4,
    'ax5': ax5,
    'ax6': ax6,
    'ax7': ax7,
    'ax8': ax8,
    'ax9': ax9,
    'ax10': ax10,
    'ax11': ax11,
    'ax12': ax12
}
# can call like: axes_dict['ax3'].hist([1, 2, 3, 4, 5])

mo_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

if moor == 'CE09OSSM':
    ymin = -550
    ymax = 5
    yticks = np.arange(ymin, ymax, 50)
    yticks = np.concatenate((yticks,ymax),axis=None)

    if vel=='U':
        xmin = -0.2
        xmax = 0.2
        xticks = np.arange(xmin, xmax+0.1, 0.1)

        xmin2 = -0.2
        xmax2 = 0.3
        xticks2 = np.arange(xmin2, xmax2, 0.1)
        xticks2 = np.concatenate((xticks2,xmax2),axis=None)

    if vel=='V':
        xmin = -0.3
        xmax = 0.3
        xticks = np.arange(xmin, xmax+0.1, 0.1)
        xticks = np.concatenate((xticks,xmax),axis=None)

        xmin2 = -0.3
        xmax2 = 0.3
        xticks2 = np.arange(xmin2, xmax2, 0.1)
        xticks2 = np.concatenate((xticks2,xmax2),axis=None)

if moor == 'CE07SHSM':
    ymin = -90
    ymax = 5
    yticks = np.arange(ymin, ymax, 10)
    yticks = np.concatenate((yticks,ymax),axis=None)

    if vel=='U':
        xmin = -0.2
        xmax = 0.2
        xticks = np.arange(xmin, xmax+0.1, 0.1)

        xmin2 = -0.2
        xmax2 = 0.2
        xticks2 = np.arange(xmin2, xmax2, 0.1)
        xticks2 = np.concatenate((xticks2,xmax2),axis=None)

    if vel=='V':
        xmin = -0.6
        xmax = 0.6
        xticks = np.arange(xmin, xmax+0.1, 0.3)
        #xticks = np.concatenate((xticks,xmax),axis=None)

        xmin2 = -0.6
        xmax2 = 0.6
        xticks2 = np.arange(xmin2, xmax2, 0.1)
        #xticks2 = np.concatenate((xticks2,xmax2),axis=None)

#####################################################################
# plot mfd ADCP , fn_mfd
picklepath = in_dir / fn_mfd  
    
if os.path.isfile(picklepath)==False:
    print('no file named: ' + fn_mfd)
    sys.exit()
    
with open(picklepath, 'rb') as fp:
    mfd = pickle.load(fp)
    print('loaded pickled mfd '+str(thisyr))

for idx in range(1,13):
    if np.any(mfd['mean'].index.month==idx):
        #std = np.array(mfd['stdev'][mfd['stdev'].index.month==idx]).squeeze() 
        mean = np.array(mfd['mean'][mfd['mean'].index.month==idx]).squeeze()
        #axes_dict[str('ax'+str(idx))].fill_between()
        axes_dict[str('ax'+str(idx))].plot(mean,mfd['z'], color = 'navy',marker = 'none', linewidth=2, alpha=0.9, label = 'ADCP mfd')

#####################################################################
# plot nsif ADCP , fn_nsif
picklepath = in_dir / fn_nsif 
    
if os.path.isfile(picklepath)==False:
    print('no file named: ' + fn_nsif)
    sys.exit()
    
with open(picklepath, 'rb') as fp:
    nsif = pickle.load(fp)
    print('loaded pickled mfd '+str(thisyr))

for idx in range(1,13):
    if np.any(nsif['mean'].index.month==idx):
        #std = np.array(mfd['stdev'][mfd['stdev'].index.month==idx]).squeeze() 
        mean = np.array(nsif['mean'][nsif['mean'].index.month==idx]).squeeze()
        #axes_dict[str('ax'+str(idx))].fill_between()
        axes_dict[str('ax'+str(idx))].plot(mean,nsif['z'], color = 'dodgerblue',marker = 'none', linewidth=2,alpha=0.9,label = 'ADCP nsif')

#####################################################################
# plot veltpa , fn_7
picklepath = in_dir / fn_7
    
if os.path.isfile(picklepath)==False:
    print('no file named: ' + fn_7)
    sys.exit()
    
with open(picklepath, 'rb') as fp:
    vel7 = pickle.load(fp)
    print('loaded pickled mfd '+str(thisyr))

for idx in range(1,13):
    if np.any(vel7['mean'].index.month==idx):
        #std = np.array(mfd['stdev'][mfd['stdev'].index.month==idx]).squeeze() 
        mean = np.array(vel7['mean'][vel7['mean'].index.month==idx]).squeeze()
        #axes_dict[str('ax'+str(idx))].fill_between()
        axes_dict[str('ax'+str(idx))].plot(mean,-7, color = 'cornflowerblue',linestyle = 'none',alpha=0.9, marker = '*', label = '-1m')

#####################################################################
# plot veltpa , fn_1
picklepath = in_dir / fn_1
    
if os.path.isfile(picklepath)==False:
    print('no file named: ' + fn_1)
    sys.exit()
    
with open(picklepath, 'rb') as fp:
    vel1 = pickle.load(fp)
    print('loaded pickled mfd '+str(thisyr))

for idx in range(1,13):
    if np.any(vel1['mean'].index.month==idx):
        #std = np.array(mfd['stdev'][mfd['stdev'].index.month==idx]).squeeze() 
        mean = np.array(vel1['mean'][vel1['mean'].index.month==idx]).squeeze()
        #axes_dict[str('ax'+str(idx))].fill_between()
        axes_dict[str('ax'+str(idx))].plot(mean,-1, color = 'lightsteelblue',linestyle = 'none',alpha=0.9, marker = '*', label = '-7m')
        axes_dict[str('ax'+str(idx))].set_ylim(ymin,ymax)
        axes_dict[str('ax'+str(idx))].set_yticks(yticks)
        axes_dict[str('ax'+str(idx))].set_xlim(xmin,xmax)
        axes_dict[str('ax'+str(idx))].set_xticks(xticks)

        ticks = axes_dict[str('ax'+str(idx))].get_yticks()
        labels = ['' if i % 2 != 0 else str(int(tick)) for i, tick in enumerate(ticks)]
        axes_dict[str('ax'+str(idx))].set_yticklabels(labels)
        axes_dict[str('ax'+str(idx))].set_title(mo_list[idx-1])

        axes_dict[str('ax'+str(idx))].axvline(x=0, color='k', linestyle=':')

        if (idx ==1) | (idx == 7):
            axes_dict[str('ax'+str(idx))].set_ylabel('Z altitudes(?) m')
        
        if idx>6: 
            if vel == 'U':
                axes_dict[str('ax'+str(idx))].set_xlabel('u m/s')
            if vel == 'V':
                axes_dict[str('ax'+str(idx))].set_xlabel('v m/s')
        
        if (idx==6) & (moor =='CE09OSSM'):
            axes_dict[str('ax'+str(idx))].set_xlim(xmin2,xmax2)
            axes_dict[str('ax'+str(idx))].set_xticks(xticks2)

#####################################################################
# plot model1  cas7_t0_x4b
picklepath = in_dir / model1
    
if os.path.isfile(picklepath)==False:
    print('no file named: ' + model1)
    sys.exit()
    
with open(picklepath, 'rb') as fp:
    x4b = pickle.load(fp)
    print('loaded pickled mfd '+str(thisyr))

for idx in range(1,13):
    m = x4b['mean'][idx-1,:]
    axes_dict[str('ax'+str(idx))].plot(m,x4b['z'], color = 'Crimson',linestyle = '-',alpha=0.9, marker = 'none', label = 'cas7_t0_x4b')

#####################################################################
# plot model1  cas7_t1_x4a
picklepath = in_dir / model2
    
if os.path.isfile(picklepath)==False:
    print('no file named: ' + model2)
    sys.exit()
    
with open(picklepath, 'rb') as fp:
    x4a = pickle.load(fp)
    print('loaded pickled mfd '+str(thisyr))

for idx in range(1,13):
    m = x4a['mean'][idx-1,:]
    axes_dict[str('ax'+str(idx))].plot(m,x4a['z'], color = 'Gold',linestyle = '-',alpha=0.9, marker = 'none', label = 'cas7_t1_x4a')

    if idx==1:
        if vel=='U':
            axes_dict[str('ax'+str(idx))].legend(loc='center right',fontsize=6,frameon=False)
        if vel=='V':
            axes_dict[str('ax'+str(idx))].legend(loc='center left',fontsize=6,frameon=False)

figname = moor+'_'+vel+'_'+str(thisyr)+'_monthly_profiles.png'
fig.savefig(out_dir / figname)
