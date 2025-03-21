"""
multi_mooring_driver.py wasn't executing unless select get_all True. 
Going to fix this, but in the interim, need to send a short list to 
A Ritger for her shell data. Going to remove the vn_dlist variables 
from each mooring extraction.

"""
# imports
from lo_tools import Lfun, zfun, zrfun

from os import listdir
from os.path import isfile, join
import xarray as xr

Ldir = Lfun.Lstart()

fn_i = Ldir['parent'] / 'LO_output' / 'extract' / 'cas7_t0_x4b' / 'moor' / 'Ritger_sites'
fn_list = [f for f in listdir(fn_i) if isfile(join(fn_i, f))]
numfiles = len(fn_list)

fn_o = Ldir['parent'] / 'LKH_output' / 'Ritger_OAH' / 'cas7_t0_x4b' / 'Ritger_sites'
Lfun.make_dir(fn_o, clean=True)

# list of vars removing from file
vn_dlist = ['Pair','Uwind','Vwind','shflux','ssflux','latent','sensible','lwrad','swrad','sustr','svstr','bustr','bvstr','u','v','w','ubar','vbar','AKs','AKv']

for ydx in range(0,numfiles): 
    fn = fn_list[ydx]
    fn_open = fn_i / fn
    ds = xr.open_dataset(fn_open, decode_times=True)

    for vn in vn_dlist:
        del ds[vn]

    fn_out = fn_o / fn
    ds.to_netcdf(fn_out, unlimited_dims='ocean_time')

    print('saved: ' + fn)




