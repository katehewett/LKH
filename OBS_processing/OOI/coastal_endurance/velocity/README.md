
# This README describes available velocity data collected at the Coastal Surface Mooorings in the OOI CE array 
 
This README and associated python code in this directory were developed to document access to OOI assets from the the [Coastal Endurance (CE) Array](https://oceanobservatories.org/array/coastal-endurance/) off the Coast of Oregon and Washington.  ***The objective was to grab and organize velocity data from the CE array's coastal moorings and organize to LO format.***

## Coastal Surface Moorings (SM) Description 
**Description:** A Surface Mooring is a type of mooring that contains a surface buoy floating on the sea surface and instruments located at fixed depths through the water column. The surface buoy provides a platform on which to secure surface instruments, allowing for the collection of data in the air and in the water, as well as an antenna to transmit data to shore via satellite. There are 6 surface moorings in the Coastal Endurance Array:
![image](https://oceanobservatories.org/wp-content/uploads/2015/09/Coastal-Endurance-Array.jpg)
***Figure 1: Coastal Endurance Array Platform Configuration (source: OOI website).*** 

***Coastal Surface Mooring List:***  
Long Name (short name) -- listed bottom-water depth m*
1. Oregon Inshore Surface Mooring [(CE01ISSM)](https://oceanobservatories.org/site/ce01issm/) -- 25m 
2. Oregon Shelf Surface Mooring [(CE02SHSM)](https://oceanobservatories.org/site/ce02shsm/) -- 80m 
3. Oregon Offshore Surface Mooring [(CE04OSSM)](https://oceanobservatories.org/site/ce04ossm/) -- 588m 

4. Washington Inshore Surface Mooring [(CE06ISSM)](https://oceanobservatories.org/site/ce06issm/) -- 29m 
5. Washington Shelf Surface Mooring [(CE07SHSM)](https://oceanobservatories.org/site/ce07shsm/) -- 87m 
6. Washington Offshore Surface Mooring [(CE09OSSM)](https://oceanobservatories.org/site/ce09ossm/) -- 542m   
  

The CE Coastal Surface Moorings generally have instrumentation mounted at the surface buoy, Near Surface Instrument Frame (NSIF) (here, 7m), and Multi-Functional Node (MFN; bottom - bottom depths listed above).  

***Naming structure:***  
* Part 1 is the Research Array ID, here CE = Coastal Endurance 

* Part 2 is the location on the shelf:  
IS = Inner Shelf  
SM = Shelf  
OS = Outer Shelf  

* Part 3 is the type of mooring:  
SM = Surface Mooring   
SP = Surface Piercing Profiler Mooring  
PM = Profiler Mooring  

* Example, CE01ISSM:  
Coastal Endurance. Site Designation 01, Oregon Line. Inner Shelf. Surface Mooring*

>**Note: The numbers (1-6) on Figure 1 mark current mooring locations, but do not represent the site designation number in each mooring name.** But rather, the Washington Line Moorings (as labeled 4-6 on Figure 1) are named CE06, CE07, and CE09, respectively. The Oregon Line Moorings (as labeled 1-3 on Figure 1) are named CE01, CE02, and CE04, respectively. CE08 and CE04 were intentionally skipped and held as a placeholders for potential mooring locations (@ ~130-150m depth). 

## Velocity data inventory:  
**WASHINGTON**  
An inventory of velocity data (+instruments +named files) are located here:  
/Users/katehewett/Documents/LKH_data/OOI/CE (personal computer)  
/dat1/kmhewett/LKH_data/OOI/CE (on apogee)

*If curious (or forget), and trying to follow naming structures of downloaded data, Appendix A provides a description of the coding system OOI teams use to track their instruments. The OOI *"Reference Designator"* system is described with links and an example.*

## Processing steps:  
**WASHINGTON**  
*KH housekeeping. v1_8June2025: an old set of code used to explore velocity data. Deleted all output files; made a new version; and started over for consistency.*

Once we have all the velcoity data from the WA OOI line [insert photo]

[make whats there plot]

Some instruments look like they have been QC'd a little more than others. This is a less rigirous QC just so that we can compare LO output with OOI velocities.   
1. **(A)** Remove spots where the Z are way off. It looks like a flag value of -999 and/or -9999 was used, but then the bins were saved instead of deleting. If this was the case, the whole WC value for that timestamp was dropped. **(B)** Remove duplicates: There are ~104k+ duplicates because of a double saving error. For a subset of time there are raw binned velocities saved and velocities saved with 10 decimal spaces. 
2. Remove 10*STD crazy outliers from each instrument type.  


3. Take out the weird profile in XXX
4. Re-bin data so we can take daily averages and process. 
5. Run a spike test 


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