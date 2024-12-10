# This README is guide for accessing moored assets from the Coastal Endurance Array

This README (and associated python code) were developed to document (and to facilitate) access to OOI assets from the the [Coastal Endurance (CE) Array](https://oceanobservatories.org/array/coastal-endurance/) off the Coast of Oregon and Washington. The code uses the OOI M2M API to access data and either loads it into the user workspace as an xarray dataset, or saves it to disk as a NetCDF file, depending on how you call it. Some examples of this are also saved in the ooi-data-explorations/python/ [repo](https://github.com/katehewett/ooi-data-explorations/tree/master/python).  

An overview of the CE Array Platforms and Instruments is provided in Table 1.  CE assets are organized across the region like this:
![image](https://oceanobservatories.org/wp-content/uploads/2015/09/Coastal-Endurance-Array.jpg)
***Figure 1: Coastal Endurance Array Platform Configuration (source: OOI website, last accessed in December 2024).***

***Table 1: Coastal Endurance Array Platforms & Instruments:***
![image](https://oceanobservatories.org/wp-content/uploads/2023/09/Endurance-Array-InstTable_2018-03-23_ver_5-00-scaled.jpg)  
*Table 1 source = OOI website, last accessed in December 2024*

***This README assumes repos are properly setup for ooi-data-explorations (seperate README provided one level back).***  
And... it's a work-in-progress. At the moment, data processing steps have been completed for the checked assets (below). Available data products are provided above in Table 1.  
** ***Mooring Assets***  
<input type="checkbox" checked> Coastal Surface Moorings   
<input type="checkbox" unchecked> Coastal Surface-Piercing Profiler Moorings   
<input type="checkbox" unchecked> Coastal Profiler Moorings

** ***Cabeled Array Assets***  
<input type="checkbox" unchecked> Cabled Deep Profiler Mooring  
<input type="checkbox" unchecked> Cabled Shallow Profiler Mooring  
<input type="checkbox" unchecked> Cabled Shallow Benthic Experiment Packages (BEPs)   

** ***Mobile Assets***  
<input type="checkbox" unchecked> Coastal Gliders  

## Setup new python environment 
<span style="background-color: #FFFF00">I setup a new env 2 ways, you only need one (I was testing something - come back to this later)</span>

1. Setup a new python environment named ooi:
    - Navigate to where you saved you ooi-data-explorations repo. For me it's here: 
    /Users/katehewett/Documents/OBS_repos/ooi-data-explorations/python  
    And do:
        ``` 
        cd python
        conda env create -f environment.yml > env.log &
        ```

2. Setup a new python environment and named it kh_ooi (similar to setting up my LO env, kh_loenv).  
    - I saved a copy of `environment.yml` on my computer. The original can be found [here](https://github.com/katehewett/ooi-data-explorations/blob/master/python/environment.yml); I saved the copy under /Users/katehewett/Documents/LKH/OBS_processing/OOI.
    - I renamed it `ooi_environment.yml`. 
    - edited the .yml file so that `name: kh_ooi` is the first line
    - I did not add or subtract packages (so far)
    - then do: `conda env create -f ooi_environment.yml > env.log &` 

>Do `conda info --envs` to find out what environments you have. And if you want to cleanly get rid of an environment, just make sure it is not active, then do `conda env remove -n myenv` or whatever name you are wanting to remove. This will not delete your yml file.

## Mooring Assets 
### Naming structure:
Part 1 is the Research Array ID:  
CE = Coastal Endurance 

Part 2 is the location on the shelf:  
IS = Inner Shelf  
SM = Shelf  
OS = Outer Shelf  

Part 3 is the type of mooring:  
SM = Surface Mooring   
SP = Surface Piercing Profiler Mooring  
PM = Profiler Mooring  

*Example, CE01ISSM:  
Coastal Endurance. Site Designation 01, Oregon Line. Inner Shelf. Surface Mooring*

>**Note: The numbers (1-6) on Figure 1 mark current mooring locations, but do not represent the site designation number in each mooring name.** But rather, the Washington Line Moorings (as labeled 4-6 on Figure 1) are named CE06, CE07, and CE09, respectively. The Oregon Line Moorings (as labeled 1-3 on Figure 1) are named CE01, CE02, and CE04, respectively. CE08 and CE04 were intentionally skipped and held as a placeholders for potential mooring locations (@ ~130-150m depth). 

*If you're new here and trying to follow, see Appendix A, for a description of the coding system OOI teams use to track their instruments. The OOI *"Reference Designator"* system is described with links and an example.*

### Coastal Surface Moorings (SM)
A Surface Mooring is a type of mooring that contains a surface buoy floating on the sea surface and instruments located at fixed depths through the water column. The surface buoy provides a platform on which to secure surface instruments, allowing for the collection of data in the air and in the water, as well as an antenna to transmit data to shore via satellite. Looking at Figure 1, there are 6 surface moorings in the Coastal Endurance Array:

*Figure 1 #. Long Name (short name) -- listed bottom-water depth m*
1. Oregon Inshore Surface Mooring [(CE01ISSM)](https://oceanobservatories.org/site/ce01issm/) -- 25m 
2. Oregon Shelf Surface Mooring [(CE02SHSM)](https://oceanobservatories.org/site/ce02shsm/) -- 80m 
3. Oregon Offshore Surface Mooring [(CE04OSSM)](https://oceanobservatories.org/site/ce04ossm/) -- 588m 

4. Washington Inshore Surface Mooring [(CE06ISSM)](https://oceanobservatories.org/site/ce06issm/) -- 29m 
5. Washington Shelf Surface Mooring [(CE07SHSM)](https://oceanobservatories.org/site/ce07shsm/) -- 87m 
6. Washington Offshore Surface Mooring [(CE09OSSM)](https://oceanobservatories.org/site/ce09ossm/) -- 542m   
  

The CE Coastal Surface Moorings generally have instrumentation mounted at the surface buoy, Near Surface Instrument Frame (NSIF) (here, 7m), and Multi-Functional Node (MFN; bottom).  

---
---

## Appendix A: OOI Glossary, Descriptions and Links:
### OOI Glossary: [OOI glossary link](https://oceanobservatories.org/glossary/)

### Reference Designators
OOI maintains over 900 instruments, to track these instruments OOI teams developed a coding system. These codes are called reference designators, and they are used to help organize the data that is collected by the OOI.

We can use Reference Designators to search for data in the Data Portal. They are also used in the Raw Data and Cruise Data archives to organize the data.

If you know the codes for the instruments you use most often, it can make it a lot easier to find the data you’re looking for in the various archives.

**General Reference Designator setup (with example):**  
Example: CE01ISSM-MFD35-04-ADCPTM000  
To interpret this code, you can break it down as follows:
* **Array** (first 2 characters), Example: CE – Coastal Endurance  
* **Site** (first 8 characters, including the two Array characters) represent the Site, or platform the instrument is deployed on. Example: CE01ISSM
* **Node** (first 14 characters) define the Node that a specific instrument is plugged into. For this example, CE01ISSM-MFD35, it’s the Seafloor Multi-Function Node (MFN) on the Coastal Endurance. See also, OOI Net NODE designator section below. 
* **Port** The two numerical digits following the Node code indicates the port on the Node that the instrument is plugged into. Example: 04 
* **Instrument Class/Series** The unique instrument code is the last 9 characters of the full Reference Designator (here, ADCPTM000). The most useful parts are the six characters immediately following the dash (here, ADCPTM) which specify the Instrument Class and Series [full list here](https://oceanobservatories.org/instruments/). In this Example: ADCPTM represents Teledyne RDI - WorkHorse Sentinel 600kHz. *[I believe, ADCPTM000 is unique to ADCPTM deployed at CE01ISSM, and so on]*.
* **Instrument / Full Reference Designator** Now we have the full Reference Designator (including the site/node prefix and the port number “-04-” that the instrument is plugged into) for an instrument that is deployed in the OOI.  

    *Note, that Reference Designators are tied to the “design” location for an instrument. This means that individual instruments (with their own serial numbers) are assigned to the same Reference Designator for the same location when they are swapped out between deployments. This, you can use a Reference Designator to retrieve multiple years of data for the same location, no matter how many actual instruments or deployments there have been.*

And a link to the more detailed [OOI description](https://oceanobservatories.org/knowledgebase/how-to-decipher-a-reference-designator/)

### OOI Net NODE designators 
For moorings, the Node is typically a controller computer & communications box, but the code can also indicate where on the mooring the Node is located. This can tell you whether a Node is on the surface buoy, mooring riser, profiler, or a multi-function node on the seafloor. Nodes that start with “SB” indicate instruments located on the Surface Buoy, while RIC and RID are nodes located on the near surface instrument frame (7 or 12 m depth for coastal and global moorings, respectively). All other “RI” codes are nodes on the mooring riser (the cable that connects the buoy to the anchor on a mooring). For profilers, WFP indicates a wire-following profiler, SF is the cabled shallow profiler, SC and PC are nodes on the cabled shallow profiler 200 m platform, and DP indicates a deep profiler.

A list of Node designators I came across while processing data:   
**[Node abbreviation] Long Name: description:**
* **["SB"] A Surface Buoy** is a type of buoy that floats on the sea surface providing buoyancy to support the mooring riser. Additionally, the surface buoy provides a platform for mounting atmospheric and ocean surface instruments and houses equipment for power generation and storage, data aggregation and recording, and two-way telemetry and GPS location. *All surface moorings contain a surface buoy.
* **["RI"] The Near-Surface Instrument Frame (or NSIF)** is an instrumented cage suspended below a surface mooring (7m for Coastal moorings, 12m for Global moorings). The NSIF contains subsurface oceanographic instruments attached to multiple data concentrator logger (DCL) computers. 
* **["MF"] Seafloor Multi-Function Nodes (MFN)** are found at the base of some surface moorings and act both as an anchor as well as a platform to affix instruments. 

>Nodes on surface moorings are labeled MFD# SBD# RID#, and not sure what the D is for, maybe data? But it seems that when it's a platform controller, they are MFC (SBC etc) and when it's an instrument collecting data, then it's MFD (SBD, etc). 

### M2M Terminology 
*Adopted text below from C.Wingard's ooi-data-explorer/python README*  

It is important to understand how requests to the OOI M2M API are structured. A request is built around the reference designator (comprised of the site, node and sensor names, see above section on reference designators), the data delivery method, and data stream (think of a stream as a data sets). Beginning and ending dates for the time period of interest are optional inputs. If omitted, all of the data for a particular instrument of interest will be downloaded.

* **Site** -- 8 character uppercase string denoting the array and location within the array of the system. These are defined on the OOI website.
* **Node** -- 5 character uppercase string (of which the first 2 characters are really the key) denoting the assembly the instrument is connected to/mounted on. These can be thought of as physical locations within/under the higher level site designator.
* **Sensor** -- 12 character uppercase string that indicates, among other things, the instrument class and series. The instrument class and series are defined on the OOI website.
* **Delivery Method** -- Method of data delivery (lowercase).  
    * ***streamed*** -- Real-time data delivery method for all cabled assets. Data is "streamed" to shore over the fiber optic network as it outputs from an instrument.  
    * ***telemetered*** -- Near real-time data delivery method for most uncabled assets. Data is recorded remotely by a data logger system and delivered in batches over a satellite or cellular network link on a recurring schedule (e.g every 2 hours).  
    * ***recovered_host*** -- Usually the same data set as telemetered for uncabled assets. Key difference is this data is downloaded from the data logger system after the asset is recovered. In most cases, this is 1:1 with the telemetered data unless there was an issue with telemetry during the deployment or the data was decimated (temporal and/or # of parameters) by the data logger system prior to transmission.  
    * ***recovered_inst*** -- Data recorded on and downloaded directly from an individual instrument after the instrument is recovered. Not all instruments internally record data, so this method will not be available for all instruments.
    * ***recovered_wfp*** -- Data recorded on and downloaded from the McLane Moored Profiler system used at several sites in OOI. Telemetered data is decimated, this data set represents the full-resolution data.
    * ***recovered_cspp*** -- Data recorded on and downloaded from the Coastal Surface Piercing Profiler system used in the Endurance array. Telemetered data is decimated, this data set represents the full-resolution data.
* **Stream** -- A collection of parameters output by an instrument or read from a file, and parsed into a named data set. Stream names are all lowercase. Streams are mostly associated with the data delivery methods and there may be more than one stream per method.

---
---
***---work in progress below --***
Notes and designations
### Coastal Surface-Piercing Profiler Moorings

Oregon Inshore Surface-Piercing Profiler Mooring(CE01ISSP)
Oregon Shelf Surface-Piercing Profiler Mooring(CE02SHSP)
Washington Inshore Surface-Piercing Profiler Mooring(CE06ISSP)
Washington Shelf Surface-Piercing Profiler Mooring(CE07SHSP)

Washington Offshore Profiler Mooring(CE09OSPM) 


**OOI Net node designator. Key terms:**
* [nsif] The Near-Surface Instrument Frame (or NSIF) is an instrumented cage suspended below a surface mooring (7m for Coastal moorings, 12m for Global moorings). The NSIF contains subsurface oceanographic instruments attached to multiple data concentrator logger (DCL) computers.
* [buoy] A Surface Buoy is a type of buoy that floats on the sea surface providing buoyancy to support the mooring riser. Additionally, the surface buoy provides a platform for mounting atmospheric and ocean surface instruments and houses equipment for power generation and storage, data aggregation and recording, and two-way telemetry and GPS location. All surface moorings contain a surface buoy.
* Seafloor Multi-Function Nodes (MFN) are found at the base of some surface moorings and act both as an anchor as well as a platform to affix instruments.
* A Profiler is a structure that moves through the water column carrying instruments, that sample across the profiler’s depth range. Profilers either track along the mooring riser (wire-following profilers), or are tethered to a mooring-mounted winch that pays out line allowing the profiler to rises through the water column until fixed depth (shallow profilers & surface piercing profilers).
* A Surface Piercing Profiler is a type of profiler that is tethered to a mooring-mounted winch at a fixed depth in the water column. As the winch pays out cable, the surface piercing profiler rises through the water column until the profiler pierces the sea surface. Instruments are affixed to the surface piercing profiler, allowing for the measurement of ocean processes at the thin surface layer.
* A Wire-Following Profiler is a type of profiler that attaches to and moves along the mooring riser over a designated depth interval. Instruments are affixed to the wire-following profiler, allowing for sampling sub-surface ocean characteristics.


