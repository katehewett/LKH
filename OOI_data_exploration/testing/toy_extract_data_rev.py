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

from ooi_data_explorations.common import list_deployments, get_deployment_dates, get_vocabulary, m2m_request, \
    m2m_collect, update_dataset, CONFIG, ENCODINGS
from ooi_data_explorations.uncabled.process_ctdbp import ctdbp_datalogger

# Setup needed parameters for the request, the user would need to vary these to suit their own needs and
# sites/instruments of interest. Site, node, sensor, stream and delivery method names can be obtained from the
# Ocean Observatories Initiative web site. The last two will set path and naming conventions to save the data
# to the local disk
site = 'CE02SHSM'           # OOI Net site designator
node = 'RID27'              # OOI Net node designator
sensor = '03-CTDBPC000'     # OOI Net sensor designator
stream = 'ctdbp_cdef_dcl_instrument'  # OOI Net stream name
method = 'telemetered'      # OOI Net data delivery method
#level = 'nsif'              # local directory name, level below site
#instrmt = 'ctdbp'           # local directory name, instrument below level

# We are after telemetered data. Determine list of deployments and use the last, presumably currently active,
# deployment to determine the start and end dates for our request.
vocab = get_vocabulary(site, node, sensor)[0]
deployments = list_deployments(site, node, sensor)
deploy = deployments[-1]
start, stop = get_deployment_dates(site, node, sensor, deploy)


