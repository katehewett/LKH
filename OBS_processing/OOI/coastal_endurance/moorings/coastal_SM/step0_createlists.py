"""
Access Coastal Surface Mooring Data 
using M2M interface 

"""

testing = True

import sys
import pprint

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
nodes