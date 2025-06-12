'''
combine instruments for LO comparisons
'''

import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta
import posixpath

from lo_tools import Lfun

import matplotlib.pyplot as plt

Ldir = Lfun.Lstart()

moor = 'CE09OSSM'
#moor = 'CE07SHSM'
#moor = 'CE06ISSM'


thisyr = 2017

loco = 'nsif'
in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco / 'daily' / 'by_year' 
fn_nsif = str(moor) + '_' + loco + '_ADCP_daily_'+str(thisyr)+'.01.01_'+str(thisyr)+'.12.31.nc'
in_nsif = posixpath.join(in_dir, fn_nsif)

loco = 'mfd'
in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco / 'daily' / 'by_year' 
fn_mfd = str(moor) + '_' + loco + '_ADCP_daily_'+str(thisyr)+'.01.01_'+str(thisyr)+'.12.31.nc'
in_mfd = posixpath.join(in_dir, fn_mfd)

nsif = xr.open_dataset(in_nsif, decode_times=True) 
mfd = xr.open_dataset(in_mfd, decode_times=True) 
