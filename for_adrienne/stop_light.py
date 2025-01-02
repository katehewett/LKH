"""
This is an example to create a 'stoplight' plot

I clipped out my path assignments and loading data lines of code
And instead hardcoded array assignments so that you can run the code
and get a plot easily. 

Based on the cumulative shelf volume for each region at Oag<1  
Make a stoplight plot for R1-R6

Overarching problems: I don't love this plot. 
Adding the yellow to green makes it not accessible for colorblind. 
 
"""
# imports
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl

'''
The next several lines and all the way to ENDHERE is hardcoded and sloppy, 
but inserted here so that you can run the code on your machine w/o needing my datafiles and tools
'''

# an array of values that you're going to color squares for
Cstack = array([[188.02650324, 158.68864122, 160.47987496, 130.28041218,
        154.12315177, 184.01335379, 186.57216704, 200.57365098,
        172.42406274, 188.13058054, 170.34006135],
       [157.60046235, 134.86980118, 132.66897676, 110.56542069,
        132.18259394, 158.53763573, 159.86101259, 160.34329837,
        140.23108329, 154.70288169, 145.33504605],
       [182.30077925, 160.62104421, 155.25138449, 133.9862183 ,
        152.37625534, 179.37455679, 179.80289848, 181.69905226,
        163.7256728 , 180.75410065, 162.81232596],
       [204.34069682, 173.49309501, 179.08043349, 152.21560763,
        169.732436  , 193.32697932, 195.09013609, 204.869801  ,
        190.53733719, 202.4507349 , 178.70996696],
       [202.5998906 , 160.66445738, 167.16258568, 141.89384433,
        153.53241196, 184.93062931, 185.85170876, 197.4583094 ,
        182.1349708 , 187.23075007, 162.17563947],
       [221.46591479, 166.3328986 , 178.92780406, 164.53921827,
        169.03589374, 194.97120093, 196.71339621, 220.91447237,
        195.43089483, 196.70703431, 169.39986457]])

# And the years they occurred 
YRstack = array([[2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021.,
        2022., 2023.],
       [2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021.,
        2022., 2023.],
       [2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021.,
        2022., 2023.],
       [2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021.,
        2022., 2023.],
       [2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021.,
        2022., 2023.],
       [2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021.,
        2022., 2023.]])

# A list of regions and years 
Rlist = {'R1','R2','R3','R4','R5','R6'}
yr_list = [year for year in range(2013,2024)]

# A lazy formation of the yaxis. So we stack region 1 (R1) on the top 
# you could also flip the yaxis in plotting - I just didn't
Y = np.ones(np.shape(Cstack))
Y[0,:] = Y[0,:]*6
Y[1,:] = Y[1,:]*5
Y[2,:] = Y[2,:]*4
Y[3,:] = Y[3,:]*3
Y[4,:] = Y[4,:]*2
Y[5,:] = Y[5,:]*1

'''ENDHERE'''

# colorbar things: set your levels and assign a colormap
# Originally took the Lt yellow to dark red from Adrienne's ppt:
# cmap = ListedColormap(['#FBE7A3', '#F9DA77', '#EF857E','#EA3324','#B02518'])      
# But changed the lightest yellow b/c can't see tell the diff w/ other yellows even w/o colorblind mask
levels = [100,130,160,190,220,250] 
cmap = ListedColormap(['LemonChiffon', '#F9DA77', '#EF857E','#EA3324','#B02518'])
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)
ax = plt.subplot2grid((2,3), (0,1), colspan=2) 

# Create the pcolormesh plot
pcm = ax.pcolormesh(YRstack, Y, Cstack, edgecolors='Grey', linewidths = 4, cmap=cmap, norm=norm)   # setting line width will change the border, grey is the border color
cbar = fig1.colorbar(pcm)
cbar.set_label('cumulative fractional volume')
ax.set_xticks(yr_list)
ax.set_yticks([1,2,3,4,5,6])
ax.set_yticklabels(['R6', 'R5', 'R4', 'R3', 'R2', 'R1'])
string_numbers = [str(num) for num in yr_list]
ax.set_xticklabels(string_numbers)
ax.set_title('Shelf OA status based on cumulative corrosive volumes per state (\u03A9 ag < 1)')

fig1.tight_layout()

'''
enter save code 
'''