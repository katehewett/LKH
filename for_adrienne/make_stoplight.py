"""
This is an example to create a 'stoplight' plot

It'll make a figure with 2 subplots. 
They both have the same data, but use different cmaps. 
Here: 
x-axis = year 
y-axis = 'region' 1-6
c-axis = random value assigned between 0 - 180 (cmin - cmax)
 
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
Generate random data which will be used to generate a "stoplight" plot
random numbers span cmin - cmax (0 - 180)
And we will be using 9 bins for the colorbar; 10 edges 
''' 
cmax = 180
cmin = 0 
numbins = 9 

# Create a random array and scale the array 
# This is the value behind each colored square 
# the dimensions here are (number regions, number years)
Cstack = np.random.rand(6, 11) 
Cstack = Cstack * cmax

# And assign some years (or whatever, these will be your x-values)
yr_list = [year for year in range(2013,2024)]
Y = np.ones(np.shape(Cstack))
YRstack = Y*yr_list
del Y

# A list of regions (or whatever, these will be your y-axis labels)
Rlist = {'R1','R2','R3','R4','R5','R6'}

# A lazy formation of y values. Here these rep regions 1-6
# We want to stack region 1 (R1) on the top (so we flipped it)
NC = np.shape(Cstack)[1] 
r = np.arange(1,7)
r = np.flipud(r)
r = r[:,np.newaxis]
Rstack = np.tile(r,(1,NC))
del r 

'''Plotting'''
# colorbar things: set your levels and assign a colormap
# original color bin request was 9, so we're just doing that here. 
levels = np.linspace(cmin,cmax,numbins+1) 
cmap_RYB = plt.get_cmap('RdYlBu', numbins)
cmap_RYG = plt.get_cmap('RdYlGn', numbins)

norm_RYB = mpl.colors.BoundaryNorm(levels, ncolors=cmap_RYB.N, clip=True)
norm_RYG = mpl.colors.BoundaryNorm(levels, ncolors=cmap_RYG.N, clip=True)

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)

axg = plt.subplot2grid((2,3), (0,1), colspan=2) 
axb = plt.subplot2grid((2,3), (1,1), colspan=2) 

# Create the pcolormesh plots (they're the same -- just using diff cmaps)
pcmg = axg.pcolormesh(YRstack, Rstack, Cstack, edgecolors='Grey', linewidths = 4, cmap=cmap_RYG, norm=norm_RYG)   
pcmb = axb.pcolormesh(YRstack, Rstack, Cstack, edgecolors='Grey', linewidths = 4, cmap=cmap_RYB, norm=norm_RYB)   

cbar_g = fig1.colorbar(pcmg)
cbar_g.set_label('RdYlGn')
axg.set_xticks(yr_list)
axg.set_yticks([1,2,3,4,5,6])
axg.set_yticklabels(['R6', 'R5', 'R4', 'R3', 'R2', 'R1'])
string_numbers = [str(num) for num in yr_list]
axg.set_xticklabels(string_numbers)
axg.set_title('Randomly assigned status with range: ' + str(cmin)+'-'+str(cmax))

cbar_b = fig1.colorbar(pcmb)
cbar_b.set_label('RdYlBu')
axb.set_xticks(yr_list)
axb.set_yticks([1,2,3,4,5,6])
axb.set_yticklabels(['R6', 'R5', 'R4', 'R3', 'R2', 'R1'])
string_numbers = [str(num) for num in yr_list]
axb.set_xticklabels(string_numbers)

fig1.tight_layout()
