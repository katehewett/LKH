"""
Step1:
Code calculates SA and SIG0 + N2 and dT/dz
saves a smaller nc-file 
(Didn't interpolate, just 
took value closest to 44.65N (LO lat_rho = 44.64590789))

Note: this code was writted to work on nc-files extracted using 
LO/extract/box/extract_box_chunks.py
which have grid info saved. If re-write can get grid info using
a history file and G, S, T = zrfun.get_basic_info(), which can then 
calc z_rho and z_w 

We are exploring the NHL_transect job here 
for cas7_t0_x4b years 2014 - 2019 
(2019 = last full year processed; 2013 skipped for now)

All years 
Time to extract data = 2.43 sec
Time to calc gsw vars = 0.01 sec

"""

import xarray as xr
from xarray import open_dataset, Dataset
import numpy as np
import gsw
from lo_tools import Lfun, zrfun
from time import time
import sys
import gsw

#plotting things
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import cmcrameri.cm as cmc

Ldir = Lfun.Lstart()

fnn = 'NHL_transect_2014.01.01_2019.12.31'
fn_in = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'box' / (fnn + '_chunks') / (fnn + '.nc')

out_dir = Ldir['parent'] / 'LKH_output'/ 'NHL' / 'explore_stratification' / fnn
Lfun.make_dir(out_dir, clean=True)

testing = False
    
# 1. Get some grid information and calc SA + SIG0
tt0 = time()
idx = 1     # we are taking the data from the lat nearest 44.65, but keep dimensions

ds = xr.open_dataset(fn_in, decode_times=False)

lat = np.expand_dims(ds.lat_rho[idx,:],axis=0) 
lon = np.expand_dims(ds.lon_rho[idx,:],axis=0)
h = np.expand_dims(ds.h.values[idx,:],axis=0)
mask_rho = np.expand_dims(ds.mask_rho[idx,:],axis=0)

z_w = np.expand_dims(ds.z_w[:,:,idx,:],axis=2) 
z_rho = np.expand_dims(ds.z_rho[idx,:],axis=2)  
tempC = np.expand_dims(ds.temp[idx,:],axis=2)  
SP = np.expand_dims(ds.temp[idx,:],axis=2)  

#zsurf = z_w[:,-1,:,:]
NT, NZ, NETA, NXI = tempC.shape
NW = z_w.shape[1]

print('Time to extract data = %0.2f sec' % (time()-tt0))
sys.stdout.flush()

# 2. Calc SA and SIG0: there's a holdup with sending gsw LARGE files - 
# it can accomodate 4D variables, but has an upper limit 
# and can be slow. Not a problem with the NHL data
tt0 = time()

P = gsw.p_from_z(z_rho,lat)
SA = gsw.SA_from_SP(SP, P, lon, lat)
CT = gsw.CT_from_pt(SA, tempC)
SIG0 = gsw.sigma0(SA,CT)   

print('Time to calc gsw vars = %0.2f sec' % (time()-tt0))
sys.stdout.flush()

#3. put in a dataset 
tt0 = time()
dsave = True
if dsave:     
    ds1 = Dataset()

    ds1['ocean_time'] = ds.ocean_time
    ds1['z_rho'] = z_rho
    ds1['z_w'] = z_w
    ds1['h'] = h
    ds1['mask_rho'] = mask_rho
    ds1['s_rho'] = ds.s_rho
    ds1['s_w'] = ds.s_w
    ds1['lat_rho'] = lat
    ds1['lon_rho'] = lon

    ds1['SIG0'] = (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),SIG0,{'units':'kg/m3 - 1000','long_name':'potential density anomaly'})
    ds1['SA'] = (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),SA,{'units':'g/kg','long_name':'Absolute Salinity'})
    ds1['CT'] = (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),CT,{'units':'degrees C','long_name':'Conservative Temperature'})

    ds1.attrs['source_file'] = fn_in
    
    fn_s = fnn + '_phys.nc'
    this_fn = out_dir / (fn_s)
    ds1.to_netcdf(this_fn)

print('Time to save = %0.2f sec' % (time()-tt0))
sys.stdout.flush()










