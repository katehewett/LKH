'''
Calc monthly stats lowpass (and avg) mooring extractions.

pickled files are saved for each oUd
'''

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
import pickle 

from lo_tools import Lfun

Ldir = Lfun.Lstart()

# processed data location
# TODO add in as args
moor = 'CE06ISSM'
#moor = 'CE07SHSM'
#moor = 'CE09OSSM'

grid = 'cas7_t1_x4a'
#grid = 'cas7_t0_x4b'

thisyr = 2017

in_dir = Ldir['parent'] / 'LO_output' / 'extract' / grid / 'moor' / 'OOI_WA_SM' 
fn = moor + '_' + str(thisyr) + '.01.01_' + str(thisyr) + '.12.31.nc'
fn_in = posixpath.join(in_dir,fn)

out_dir = Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / 'monthly_stats'
#fn_out = posixpath.join(out_dir, (moor+ '_monthly_' +str(thisyr)+ '_' + grid + '.nc'))

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

ds = xr.open_dataset(fn_in, decode_times=True)  
ot = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')

NT = np.shape(ds.u)[0]
NZ = np.shape(ds.u)[1]
z = np.nanmean(ds.z_rho,axis=0)
zstring = [str(number) for number in range(0, 30)]

vn_list = ['u', 'v'] # w isn't in new avg output -- deal w/ later

DTmax = 5

##########################################################################
# df2.iloc[np.where((df2.index.month==10)&(df2.index.day==2|3))]
# Calc Monthly Stats U 
for vn in vn_list:
    df = pd.DataFrame({'ocean_time':ot}) 
    df = df.set_index('ocean_time')
    df[zstring] = ds[str(vn)].values
    pmean = df.resample('M').mean()
    pstdev = df.resample('M').std()

    stats = {}
    stats['z'] = z
    stats['months'] = np.arange(1,13,1)
    stats['mean'] = np.array(pmean)
    #stats['p75'] = p75
    #stats['p25'] = p25
    stats['stdev'] = np.array(pstdev)
    stats['interval'] = 'monthly'
    stats['location'] = moor 
    stats['component'] = vn
    stats['units'] = 'm/s'
    stats['year'] = thisyr
    stats['grid'] = grid

    if vn == 'u':
        pkl = moor+ '_Umonthly_' +str(thisyr)+ '_' + grid + '.pkl'
    if vn == 'v':
        pkl = moor+ '_Vmonthly_' +str(thisyr)+ '_' + grid + '.pkl'
    if vn == 'w':
        pkl = moor+ '_Wmonthly_' +str(thisyr)+ '_' + grid + '.pkl'

    picklepath = out_dir/pkl
    with open(picklepath, 'wb') as fm:
        pickle.dump(stats, fm)  
        print('Pickled '+vn+' monthly ' + moor + ': ' + grid)
        sys.stdout.flush()