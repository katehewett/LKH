# testing script to request data from OOI M2M 
# takes ~30sec - 1 minute to run 5days, but the saving step is getting a error msg
# took ~5 minutes fo the spring - fall < but still error code when saving 

from time import time as ttime
tt0 = ttime()

import os
import sys 
import xarray as xr
import netCDF4 as nc
import numpy as np

from xarray import Dataset
from  ooi_data_explorations.common import m2m_request, m2m_collect

## setup # testing script to request data from the pH sensor on the Oregon
## Shelf Surface Mooring near-surface (7 m depth) instrument frame (NSIF).
# https://oceanobservatories.org/site/ce02shsm/: CE02SHSM-RID26-06-PHSEND000	
#site = 'CE02SHSM'           # OOI Net site designator
#node = 'RID26'              # OOI Net node designator
#sensor = '06-PHSEND000'     # OOI Net sensor designator
#method = 'telemetered'      # OOI Net data delivery method
#stream = 'phsen_abcdef_dcl_instrument'  # OOI Net stream name
#start = '2019-04-01T00:00:00.000Z'  # data for spring 2019 ...
##stop = '2019-09-30T23:59:59.999Z'   # ... through the beginning of fall
#stop = '2019-04-05T23:59:59.999Z'   # do less

# setup # testing script to request data from the CTD sensor on the Oregon
# Shelf Surface Mooring near-surface (7 m depth) 
# https://oceanobservatories.org/site/ce02shsm/: CE02SHSM-RID27-03-CTDBPC000
site = 'CE02SHSM'           # OOI Net site designator 
node = 'RID27'              # OOI Net node designator
sensor = '03-CTDBPC000'     # OOI Net sensor designator 
method = 'telemetered'      # OOI Net data delivery method
stream = 'ctdbp_cdef_dcl_instrument'  # OOI Net stream name
start = '2019-04-01T00:00:00.000Z'  # data for spring 2019 ...
#stop = '2019-09-30T23:59:59.999Z'   # ... through the beginning of fall
stop = '2019-04-05T23:59:59.999Z'   # do less

# Request the data (this may take some time)
r = m2m_request(site, node, sensor, method, stream, start, stop)

# Use a regex tag to download only the sensor data from the THREDDS catalog
# created by our request.
tag = '.*CTD.*\\.nc$'
ds = m2m_collect(r, tag)

print('Time to run toy extraction = %0.2f sec' % (ttime()-tt0))

# simple define output path and make fancy later 
out_path = '/Users/katehewett/Documents/LKH_output/OOI_data_exploration/testing'

# then output file 
out_file = ('%s_%s_%s_%s.nc' % (site, node, method, stream))
#nc_out = os.path.join(out_path, out_file)




#ds.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf',unlimited_dims='time') 


# not sure why getting error code  
#data.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf') 
# TypeError: Invalid value for attr '_FillValue': b'e'. For serialization to netCDF files, its value must be of one of the following types: str, Number, ndarray, number, list, tuple





## this is very messy - just saving TS time until figure out the error code listed above??
#ocean_time = ds.time
#SP = ds.sea_water_practical_salinity
#T = ds.sea_water_temperature

#ds1 = Dataset()
#ds1['ocean_time'] = ocean_time
#ds1['SP'] = SP
#ds1['temperature'] = T

#ds.to_netcdf(nc_out, mode='w',, format='NETCDF4', engine='h5netcdf',unlimited_dims='time') 
#ds1.to_netcdf(nc_out, unlimited_dims='ocean_time')



list(ds.keys())
list(ds.coords)
# how do i just list the attributes? 





