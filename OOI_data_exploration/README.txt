
# README for getting started with the OOI Data Explorations repository.
  
  This covers initial steps to access OOI data using M2M & scripts from OOI data explorations, here focused on python
  Scripts and readme's found here: 
  (https://github.com/oceanobservatories/ooi-data-explorations)
  
#fork/permission issues
  Under the OOI readme the step to add the upstream feed for the master repository wasn't working for me...I kept getting the following error msg, after doing: 
      git fetch upstream
	  
      git@github.com: Permission denied (publickey).
      fatal: Could not read from remote repository.
 
      Please make sure you have the correct access rights
      and the repository exists.

[ possible solution ? start] 
On github.com, I created a new fork (katehewett/ooi-data-explorations) while on https://github.com/oceanobservatories/ooi-data-explorations
^ Question: How do I fork a directory to a different level under my github? i.e. not directly under katehewett? 

On my laptop clone a copy of the fork: 

	cd ~/Documents/temp_repos
	git clone https://github.com/katehewett/ooi-data-explorations.git
	cd ooi-data-explorations/
	
	git remote add upstream https://github.com/oceanobservatories/ooi-data-explorations.git
	git fetch upstream

* For the time being, I am just moving on because not editing their scripts - still getting error msg after git fetch upstream / re: permissions. 

* Note the repository is on my laptop under "temp_repos", which is just a folder that I save repositories I don't want on LKH 

[ possible solution ? end] 

	cd ooi-data-explorations/python

# ooi-data-explorations for python 
## setup
Since I have git+miniconda set up and access credentails at OOI, I skipped down to "access credentials"
If you have a OOI login, and you login here:(https://ooinet.oceanobservatories.org/) [last accessed on June 2023]. On the map you'll see your login email in the upper right hand side of a map. Click it / select user profile and grab your login (API username) and password (API token). Note if you refresh your API token, then you need to also select submit and then close. 

My code looks like:

	cd ~
	touch .netrc
	chmod 600 .netrc
	cat <<EOT >> .netrc
	machine ooinet.oceanobservatories.org
    	login my-API-username 
    	password my-API-token
	EOT

The next step requires making an environment on your laptop. I made a copy of their file "environment.yml", and saved it as "ooi_environment.yml", under LKH/OOI_data_exploration
and do: 

	cd ~/Documents/LKH/OOI_data_exploration/
	conda env create -f ooi_environment.yml

(it only took a few mintues to create the environment); and then check:
 
    conda env list 
	
activate: 

	conda activate ooi
		
And install as a local development package: 

	cd ~/Documents/temp_repos/ooi-data-explorations/python
	conda develop .
	
## toy code and usage
See readme for description of M2M terminology and scripts. 
After completing the steps above, with "ooi_env" activated, can run a toy code to request 5 days of data from the pH sensor on the Oregon Shelf Surface Mooring near-surface (7 m depth) instrument frame (NSIF). path: ~/Documents/LKH/OOI_data_exploration/testing/toy_extract_data.py

to run a data request, we run m2m_request.py which needs parameters: site, node, sensor, stream and method. 

An OOI glossary can be found here https://oceanobservatories.org/glossary/ ; and for these scripts helpful terms are defined with listed options below. 

### OOI Sites
site = OOI net site designator: 
A comprehensive list of OOI assets can be found here:(https://oceanobservatories.org/site-list/).

We're pulling data from the Coastal Endurance Array off the coast of Oregon and Washington: (https://oceanobservatories.org/array/coastal-endurance/). 

And, a list of mooring sites for the Coastal Endurance Array: 
Oregon Inshore Surface Mooring(CE01ISSM)
Oregon Shelf Surface Mooring(CE02SHSM)
Oregon Offshore Surface Mooring(CE04OSSM)
Washington Inshore Surface Mooring(CE06ISSM)
Washington Shelf Surface Mooring(CE07SHSM)
Washington Offshore Surface Mooring(CE09OSSM)

Oregon Inshore Surface Piercing Profiler Mooring(CE01ISSP)
Oregon Shelf Surface Piercing Profiler Mooring(CE02SHSP)
Washington Inshore Surface Piercing Profiler Mooring(CE06ISSP)
Washington Shelf Surface Piercing Profiler Mooring(CE07SHSP)
Washington Offshore Profiler Mooring(CE09OSPM) 

### OOI Nodes 
A node is a section of a platform that contains one or more computers and power converters. Instruments on a platform are plugged into a node, which collects the instrument data internally and/or transmit the data externally. Some platforms contain a single node, like a glider. Other platforms have several nodes wired together. For example, a mooring that hosts a surface buoy, near-surface instrument frame, and seafloor multi-function node, each with a different set of instruments attached.

OOI Net node designator 
* [nsif] The Near-Surface Instrument Frame (or NSIF) is an instrumented cage suspended below a surface mooring (7m for Coastal moorings, 12m for Global moorings). The NSIF contains subsurface oceanographic instruments attached to multiple data concentrator logger (DCL) computers.
* [buoy] A Surface Buoy is a type of buoy that floats on the sea surface providing buoyancy to support the mooring riser. Additionally, the surface buoy provides a platform for mounting atmospheric and ocean surface instruments and houses equipment for power generation and storage, data aggregation and recording, and two-way telemetry and GPS location. All surface moorings contain a surface buoy.
* Seafloor Multi-Function Nodes (MFN) are found at the base of some surface moorings and act both as an anchor as well as a platform to affix instruments.
* A Profiler is a structure that moves through the water column carrying instruments, that sample across the profiler’s depth range. Profilers either track along the mooring riser (wire-following profilers), or are tethered to a mooring-mounted winch that pays out line allowing the profiler to rises through the water column until fixed depth (shallow profilers & surface piercing profilers).
* A Surface Piercing Profiler is a type of profiler that is tethered to a mooring-mounted winch at a fixed depth in the water column. As the winch pays out cable, the surface piercing profiler rises through the water column until the profiler pierces the sea surface. Instruments are affixed to the surface piercing profiler, allowing for the measurement of ocean processes at the thin surface layer.
* A Wire-Following Profiler is a type of profiler that attaches to and moves along the mooring riser over a designated depth interval. Instruments are affixed to the wire-following profiler, allowing for sampling sub-surface ocean characteristics.

### Instrument Classes
[**ADCPT**](https://oceanobservatories.org/instrument-series/adcpta/) - Teledyne RDI - WorkHorse  
[**CTDBP**](https://oceanobservatories.org/instrument-series/ctdbpc/) - SBE - 16plusV2  
[**DOSTA**](https://oceanobservatories.org/instrument-series/dostad/) - Aanderaa - Optode 4831  
[**FLORT**](https://oceanobservatories.org/instrument-series/flortd/) - WET Labs - ECO Triplet-w  
[**METBK**](https://oceanobservatories.org/instrument-series/metbka/) - Star Engineering - ASIMET  
[**NUTNR**](https://oceanobservatories.org/instrument-series/nutnrb/) - SBE - SUNA V2  
[**OPTAA**](https://oceanobservatories.org/instrument-series/optaad/) - SBE - AC-S  
[**PCO2A**](https://oceanobservatories.org/instrument-series/pco2aa/) - Pro-Oceanus - pCO2-pro  
[**PCO2W**](https://oceanobservatories.org/instrument-series/pco2wb/) - Sunburst - SAMI-pCO2  
[**PHSEN**](https://oceanobservatories.org/instrument-series/phsend/) - Sunburst - SAMI-pH  
[**SPKIR**](https://oceanobservatories.org/instrument-series/spkirb/) - SBE - OCR507  
[**VELPT**](https://oceanobservatories.org/instrument-series/velpta/) - Nortek - Aquadopp  
[**WAVSS**](https://oceanobservatories.org/instrument-series/wavssa/) - Axys Technologies - TRIAXYS








