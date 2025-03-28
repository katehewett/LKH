
# This README provides an intro to steps I took before accessing OOI data 

This README (and associated python code in this directory) were developed to document (and to facilitate) access to OOI assets from the the [Coastal Endurance (CE) Array](https://oceanobservatories.org/array/coastal-endurance/) off the Coast of Oregon and Washington. The end goal is to have all the data in LO format. 

The code uses the OOI M2M API to access data and either loads it into the user workspace as an xarray dataset, or saves it to disk as a NetCDF file, depending on how you call it.  

To get started you need to have (1) an ooi environment and (2) login creds. 
Steps are provided below.  

>***If this is your first time accessing OOI data / using the OOI data explorations***, NEWBIE_README.md provides instructions to help with setting ooi login credentials, and setting up the ooi data explorations repo on your machine/github. 

Folders in this directory are organized by asset type. And this is a work-in-progress. At the moment, data processing steps have been completed for the checked assets (below) for the CE array:

** ***Mooring Assets***  
<input type="checkbox" checked> Coastal Surface Moorings   
<input type="checkbox" unchecked> Coastal Surface-Piercing Profiler Moorings   
<input type="checkbox" unchecked> Coastal Profiler Moorings

** ***Cabeled Array Assets***  
<input type="checkbox" unchecked> Cabled Deep Profiler Mooring  
<input type="checkbox" unchecked> Cabled Shallow Profiler Mooring  
<input type="checkbox" unchecked> Cabled Shallow Benthic Experiment Packages (BEPs)   

## STEP 1: Create an (ooi) environment 
Create an environment that has all the modules required to interact with the ooi data exploration repo. One way to do this would be to:  
- copy the ooi [environment.yml](https://github.com/oceanobservatories/ooi-data-explorations/tree/master/python) to your machine. Rename it, for example, to ooi_environment.yml
- ***Or if you use LO tools,*** you can copy Kate's ooi environment which let's you use LO tools while working with ooi data explorations. <span style="color: red; font-weight: bold;">[INSERT LINK]
- edit the environment so that kh_ooi is on the first line (or whatever you want to call it)
- add or subtract any packages you like
- if you are using the lo_tools local package, change the path for it in the yml to be -e ../../../LO/lo_tools/ lo_tools (assuming that is the correct relative path) 
- then do: `conda env create -f ooi_environment.yml > env.log`

At any time you can do `conda info --envs` to find out what environments you have. And if you want to cleanly get rid of an environment, just make sure it is not active, then do `conda env remove -n myenv` or whatever name you are wanting to remove. This will not delete your yml file. 

## STEP 2: Grab your OOI login credentials 
Login here: [OOI site](https://ooinet.oceanobservatories.org/)  

- Just above the map, upper right hand side, you'll see a login button (or your email if you're logged in).  

- Once logged in, click your email / 'user profile' / and grab your login (API username) and password (API token).  

- Note if you refresh your API token, then you need to also select submit and then close.  

See NEWBIE_README.md if you don't have login creds.  

<span style="color: Blue; font-weight: bold;">**Don't put your username/token in places that post to your public git**







------------
<span style="color: red; font-weight: bold;">***KMH TODOs***:  
<input type="checkbox" unchecked> <span style="font-weight: normal;">Insert link  
<input type="checkbox" unchecked> Look at ooi env and check for adjustments. Right now, it's just a combination where I added on loenv dependencies and the path to install lo_tools so we can use functions when interacting with ooi data 