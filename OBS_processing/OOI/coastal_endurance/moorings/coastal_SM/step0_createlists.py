"""
Access Coastal Surface Mooring Data 
using M2M interface 

"""

testing = True

import sys
import pprint
import numpy as np

# if set up toolbox can add, but for now add paths -- fix so can work on apogee/perigee 
sys.path.append('/Users/katehewett/Documents/OBS_repos/ooi-data-explorations/python/')
sys.path.append('/Users/katehewett/Documents/OBS_repos/ooi-data-explorations/python/ooi_data_explorations.common')

# import the functions used to list available information about sites, nodes and sensors ...
# you could list sites before list_nodes, but we know which sites we want to look at and don't need ALL of the OOI sites
from ooi_data_explorations.common import list_nodes, list_sensors

# ... and the more detailed information about a specific site, node and sensor (reference designator)
from ooi_data_explorations.common import list_methods, list_streams, list_metadata

# our sites -- surface moorings (SM)
sn_name_dict = {
    'CE01ISSM':'Oregon Inshore Surface Mooring 25m',
    'CE02SHSM':'Oregon Shelf Surface Mooring 80m',
    'CE04OSSM':'Oregon Offshore Surface Mooring 588m',
    'CE06ISSM':'Washington Inshore Surface Mooring 29m',
    'CE07SHSM':'Washington Shelf Surface Mooring 87m',
    'CE09OSSM':'Washington Offshore Surface Mooring 542m',
}

site_list = list(sn_name_dict.keys())

# select CE01ISSM as the site to use for the testing example
if testing:
    site = site_list[0]

# create a list of nodes available for a particular site
nodes = list_nodes(site)
print(nodes)

if testing:
    node = nodes[2]

sensors = list_sensors(site, node)
print(sensors)

# find where CTD is at 
if testing:
    substring = 'CTD'
    #print(substring)
    for idx, item in enumerate(sensors):
        if substring in item:
            print('The instrument '+substring+' is at idx: '+str(idx))
            print(sensors[idx])
            sensor = sensors[idx]
        else:
            print(sensors[idx])
else: print('not testing')

# the steps above are required to create a reference designator that will be used to request data 
# you can do it manually, but it is worse 
methods = list_methods(site, node, sensor)
print(methods)

# select the one stream for this data delivery method to use 
if testing:
    method = methods[1] 
    streams = list_streams(site, node, sensor, method)

if testing:
    stream = list_streams(site, node, sensor,methods[1])
else:
    print('pick stream')

# create lists of dictionaries with the available parameters and time ranges covered by each 
# dataset associated with the sensor
metadata = list_metadata(site,node,sensor)

parameters = metadata.pop('parameters')
time_ranges = metadata['times']

for p in parameters:
    if p['stream'] == stream:
        pp.pprint(p)


#idx = sensors.index(target_string)
    
    

