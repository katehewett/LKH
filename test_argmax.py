"""
argmax w/ take_along_axis example 
I used a history file 
takes ~3-5 seconds on personal mac to run this code

"""
import xarray as xr
import numpy as np
from lo_tools import Lfun, zrfun

Ldir = Lfun.Lstart()
fn = Ldir['roms_out'] / 'cas6_v0_live' / 'f2022.08.08' / 'ocean_his_0021.nc'

ds = xr.open_dataset(fn, decode_times=True)
h = ds.h.values
G, S, T = zrfun.get_basic_info(fn)
z_rho, z_w = zrfun.get_z(h, 0*h, S) 

# 1 set a random condition (temp < 9) 
# 2 use arg_max to find index of first false
# 3 use take along axis to find the associated z_rho 
# In this example, I squeezed the time dimension on temp, 
# but I could have also expanded z_rho at axis=0 so it had a time
# dimension. Lesson: keep dimensions for easy use of take_along_axis
temp = ds.temp.values.squeeze()  
Tbool = temp<9    
A = np.argmax(Tbool==False,axis=0,keepdims = True)    
ztest = np.take_along_axis(z_rho,A,axis=0)

# setting keep dims = True made grabbing the z variables easier 