'''
This code: 
1) calculates z_rho z_w 
2) grabs the DX and DY for area calcs
3) calculates density and omega and saves the files 

!! code assumes extracted full years were saved XXXX.01.01 - XXXX.12.31 !! 

notes: (should have saved pn pm in job list for DX DY ease)
'''

# imports
from lo_tools import Lfun, zfun, zrfun

import os
import sys
import argparse
import xarray as xr
from time import time
import numpy as np
import pandas as pd

import gsw
import PyCO2SYS as pyco2

#import cmocean


tt0 = time()

# command line arugments
parser = argparse.ArgumentParser()
# which run was used:
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
#parser.add_argument('-ro', '--roms_out_num', type=int) # 2 = Ldir['roms_out2'], etc.
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
#parser.add_argument('-y1', '--ys1', type=str) # e.g. 2015
parser.add_argument('-job', '--job_type', type=str) # job 
# these flags get only surface or bottom fields if True
# - cannot have both True -
parser.add_argument('-surf', default=False, type=Lfun.boolean_string)
parser.add_argument('-bot', default=False, type=Lfun.boolean_string)
# Optional: set max number of subprocesses to run at any time
parser.add_argument('-Nproc', type=int, default=10)
# Optional: for testing
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
# get the args and put into Ldir
args = parser.parse_args()
# test that main required arguments were provided
argsd = args.__dict__
for a in ['gtagex']:
    if argsd[a] == None:
        print('*** Missing required argument: ' + a)
        sys.exit()
gridname, tag, ex_name = args.gtagex.split('_')
# get the dict Ldir
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
# add more entries to Ldir
for a in argsd.keys():
    if a not in Ldir.keys():
        Ldir[a] = argsd[a]

Ldir = Lfun.Lstart()

# check for input conflicts:
if args.surf==True and args.bot==True:
    print('Error: cannot have surf and bot both True.')
    sys.exit()
    
# load a grid for the z_coord calc
if args.gtagex=='cas7_t0_x4b':
    fn_hist = Ldir['roms_out'] / args.gtagex / 'f2017.12.13' / 'ocean_his_0002.nc'
    GG, SG, TG = zrfun.get_basic_info(fn_hist)
    Vtransform = SG['Vtransform']
else: 
    print('error: check grid')
    sys.exit()

thisYR = int(args.ys0)

# name the output file where files will be dumped
fn_o = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / args.job_type
fn_n = args.job_type+'_wARAG_'+str(thisYR)+'.01.01_'+str(thisYR)+'.12.31'+'.nc'
fn_out = fn_o / fn_n
Lfun.make_dir(fn_o, clean=False)

# and input file location
fna = args.job_type+'_'+str(thisYR)+'.01.01_'+str(thisYR)+'.12.31_chunks' 
fnb = args.job_type+'_'+str(thisYR)+'.01.01_'+str(thisYR)+'.12.31'+'.nc'
fn_i = Ldir['LOo'] / 'extract' / args.gtagex / 'box' / fna
fn_in = fn_i / fnb

if os.path.isfile(fn_in)==False:
    print('no file named: '+fnb)
    sys.exit()

# load file 
ds = xr.open_dataset(fn_in, decode_times=True)    
NT,NR,NC = np.shape(ds.salt.values)

if np.any(ds.mask_rho.values==0)==True:
    print('weird - masked values')
    sys.exit()

# calculate zrho and zw: 
# reassign (box chunks doens't save Vtransform and N)
S = {}
S['s_rho'] = ds.s_rho.values
S['s_w'] = ds.s_w.values
S['hc'] = ds.hc.values
S['Cs_w'] = ds.Cs_w.values
S['Cs_r'] = ds.Cs_r.values
S['N'] = np.shape(ds.salt)[1]
S['Vtransform'] = np.array(2)#, dtype=int32)
h = ds['h'].values
zeta = ds.zeta.values

z_rho = np.full([NT,NR,NC],np.nan)
z_w = np.full([NT,NR+1,NC],np.nan)
for idx in range(0,NT):
    Z = zeta[idx,:].squeeze()
    ZR, ZW = zrfun.get_z(h, Z, S)
    z_rho[idx,:]=ZR
    z_w[idx,:]=ZW

# grab DX DY (should have saved pn pm)
Lon = GG['lon_rho'][0,:]
Lat = GG['lat_rho'][:,0]
lon = ds['lon_rho'].values
lat = ds['lat_rho'].values

ilon0 = zfun.find_nearest_ind(Lon, lon[0])
ilon1 = zfun.find_nearest_ind(Lon, lon[-1])
ilat0 = zfun.find_nearest_ind(Lat, lat[0])
ilat1 = zfun.find_nearest_ind(Lat, lat[-1])

DX = GG['DX'][ilat1,ilon0:ilon1+1]
DY = GG['DY'][ilat1,ilon0:ilon1+1]
# GG['lon_rho'][ilat1,ilon0:ilon1+1]
# GG['lat_rho'][ilat1,ilon0:ilon1+1]

# Calc rho and prepare for variables for carbon calculations 
ot = pd.to_datetime(ds.ocean_time.values)

SP = ds.salt.values          # practical salinity
SP[SP<0] = 0                 # Make sure salinity is not negative. Could be a problem for pyco2.
    
PT = ds.temp.values          # potential temperature [degC] 

Pres = gsw.p_from_z(z_rho,lat[0])

SA = gsw.SA_from_SP(SP, Pres, lon[0], lat[0]) # absolute salinity [g kg-1]
CT = gsw.CT_from_pt(SA, PT) # conservative temperature [degC]
rho = gsw.rho(SA, CT, Pres) # in situ density [kg m-3]
ti = gsw.t_from_CT(SA, CT, Pres) # in situ temperature [degC]

ALK = ds.alkalinity.values   # alkalinity [milli equivalents m-3 = micro equivalents L-1]
TIC = ds.TIC.values          # TIC [millimol C m-3 = micromol C L-1]     
# Convert from micromol/L to micromol/kg using in situ dentity because these are the
# units expected by pyco2.
ALK1 = 1000 * ALK / rho
TIC1 = 1000 * TIC / rho

# I'm not sure if this is needed. In the past a few small values of these variables had
# caused big slowdowns in the MATLAB version of CO2SYS.
ALK1[ALK1 < 100] = 100
TIC1[TIC1 < 100] = 100

 # 3. Calculate aragonite saturation 
CO2dict = pyco2.sys(par1=ALK1, par1_type=1, par2=TIC1, par2_type=2,
    salinity=SP, temperature=ti, pressure=Pres,
    total_silicate=50, total_phosphate=2, opt_k_carbonic=10, opt_buffers_mode=0)
cARAG = CO2dict['saturation_aragonite']

ds['ARAG'] = (('ocean_time', 's_rho', 'xi_rho'),cARAG,{'units': ' '})
ds['ARAG'].attrs['long_name'] = 'aragonite saturation state'

ds['SP'] = (('ocean_time', 's_rho', 'xi_rho'),SP,{'units': ' '})
ds['SP'].attrs['long_name'] = 'Practical Salinity'

ds['SA'] = (('ocean_time', 's_rho', 'xi_rho'),SA,{'units': 'g kg-1'})
ds['SA'].attrs['long_name'] = 'Absolute Salinity'

ds['CT'] = (('ocean_time', 's_rho', 'xi_rho'),CT,{'units': 'degC'})
ds['CT'].attrs['long_name'] = 'Absolute Salinity'

ds['rho'] = (('ocean_time', 's_rho', 'xi_rho'),rho,{'units': 'kg m-3'})
ds['rho'].attrs['long_name'] = 'in situ density kg m-3'

ds['z_rho'] = (('ocean_time', 's_rho', 'xi_rho'),z_rho,{'units': 'm'})
ds['z_rho'].attrs['long name'] = 'vertical position on s_rho grid, positive up'

ds['z_w'] = (('ocean_time', 's_w', 'xi_rho'),z_w,{'units': 'm'})
ds['z_w'].attrs['long name'] = 'vertical position on s_w grid, positive up'

ds.to_netcdf(fn_out, unlimited_dims='ocean_time')