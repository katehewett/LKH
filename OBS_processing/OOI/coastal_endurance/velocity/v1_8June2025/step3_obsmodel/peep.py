'''
plot whats there . 
'''


import sys
import os
import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

moor = 'CE09OSSM'

mfd = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/mfd/daily/'+moor+'_mfd_ADCP_DAILY.nc')

if (moor == 'CE09OSSM') | (moor == 'CE07SHSM'):
    nsif = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/nsif/daily/'+moor+'_nsif_ADCP_DAILY.nc')

VEL7 = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/nsif/daily/'+moor+'_nsif_VELPTA_DAILY.nc')
VEL1 = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/'+moor+'/velocity/surfacebuoy/daily/'+moor+'_surfacebuoy_VELPTA_DAILY.nc')


t1V1 = VEL1.ocean_time[0]
t2V1 = VEL1.ocean_time[-1]

t1V7 = VEL7.ocean_time[0]
t2V7 = VEL7.ocean_time[-1]

t1mfd = mfd.ocean_time[0]
t2mfd = mfd.ocean_time[-1]

t1nsif = nsif.ocean_time[0]
t2nsif = nsif.ocean_time[-1]

print('V1 range: ' + t1V + ' ' + t2V1)
print('V7 range: ' + t1V7 + ' ' + t2V7)
print('mfd range: ' + t1mfd+ ' ' + t2mfd)
print('nsif range: ' + t1nsif + ' ' + t2nsif)


plt.plot(mfd.u.values[10,:],mfd.z.values)
plt.plot(nsif.u.values[10,:],nsif.z.values)
