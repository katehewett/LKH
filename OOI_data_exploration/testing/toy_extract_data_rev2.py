import requests
import time
from thredds_crawler.crawl import Crawl
import os
import xarray as xr
import matplotlib.pyplot as plt

USERNAME = 'kmhewett@uw.edu'
TOKEN= 'mcwZ9u9DTAsfGha'
DATA_API_BASE_URL = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'

data_request_url = DATA_API_BASE_URL+\
                    'RS03ASHS/'+\
                    'MJ03B/'+\
                    '07-TMPSFA301/'+\
                    'streamed/'+\
                    'tmpsf_sample'+'?'+\
                    'beginDT=2017-09-04T17:54:58.050Z&'+\
                    'endDT=2017-09-11T23:54:58.050Z'

r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
data = r.json()

print(data['allURLs'][0])

print(data['allURLs'][1])

check_complete = data['allURLs'][1] + '/status.txt'
for i in range(1000): 
    r = requests.get(check_complete)
    if r.status_code == requests.codes.ok:
        print('request completed')
        break
    else:
        time.sleep(.5)  


url = data['allURLs'][0]
url = url.replace('.html', '.xml')
tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC'
c = Crawl(url, select=[".*\.nc$"], debug=False)
datasets = [os.path.join(tds_url, x.id) for x in c.datasets]
splitter = url.split('/')[-2].split('-')
dataset_url = datasets[0]


ds = xr.open_dataset(dataset_url)
ds = ds.swap_dims({'obs': 'time'})
ds['temperature04'].plot.line()
plt.ylabel('Temperature in degrees Celsius')
plt.xlabel('Time')
plt.legend()
plt.show()



