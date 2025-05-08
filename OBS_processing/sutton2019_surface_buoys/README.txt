# OBS PROCESSING: 
## Sutton et al 2019 Surface buoy data 

This code was created to pull and then process surface data from NOAA data products for 5 buoys:
    Chaba buoy : https://www.pmel.noaa.gov/co2/timeseries/CHABA.txt
    Cape Elizabeth buoy : https://www.pmel.noaa.gov/co2/timeseries/CAPEELIZABETH.txt
    Cape Arago buoy : https://www.pmel.noaa.gov/co2/timeseries/CAPEARAGO.txt
    CCE2 buoy : https://www.pmel.noaa.gov/co2/timeseries/CCE2.txt
    CCE1 buoy : https://www.pmel.noaa.gov/co2/timeseries/CCE1.txt

Notes:
	1- Extracted 5 buoy datasets. But only 3 are within the LO domain (CHABA, CE, CA)
	2- Data sources listed at end, and additional details provided in each text file listed above. 
	3- Text files are also saved under: /LKH_data/sutton2019_surface_buoys/data_files/

Flow of scripts:
process_webdata.py: run to process files, which will read the textfiles and save:
/LKH_data/sutton2019_surface_buoys/py_files/

make_dailyavg.py: run to create daily averages of mooring data and save:
/LKH_data/sutton2019_surface_buoys/daily/

**Date of last data download: 5 September 2024**
**Process date 6 September 2024 (again on 11 Sept 2024)** 

###Variables:
datetime_utc: Date and time of measurement in UTC (YYYY-MM-DD HH:MM)
SST: Sea surface temperature at 0.5m depth (degrees C); uncertainty <0.01
SSS: Sea surface salinity at 0.5m depth; uncertainty <0.05
pCO2_sw: seawater pCO2 at <0.5m depth (µatm); uncertainty <2
pCO2_air: air pCO2 at 0.5-1m height (µatm); uncertainty <1
xCO2_air: air xCO2 at 0.5-1m height (ppm); uncertainty <1
pH_sw: seawater pH at 0.5m depth; uncertainty <0.02
DOXY: salinity-compensated dissolved oxygen at 0.5m depth (µmol kg-1); manufacturer-stated uncertainty <5%
CHL: fluorescence-based nighttime chlorophyll-a at 0.5m depth (µg l-1) adjusted with the community-established
calibration bias of 2 (https://doi.org/10.1002/lom3.10185); manufacturer-stated resolution <0.025
NTU: turbidity at 0.5m depth (NTU); manufacturer-stated resolution <0.013

Notes on QC: 
* The time series data saved in .txt files, at paths listed above, only include data from the original deployment-level data files that were assigned QF = 2 ("good data"). 
* Any missing values or values assigned QF of 3 or 4 in the original deployment-level data are replaced with "NAN" in the time series product that we pulled. And then missing data (mostly under NTU) are also saved as "NAN".
* Original source: All post-calibrated and quality-controlled data are archived at NCEI:  https://www.ncei.noaa.gov/access/ocean-carbon-acidification-data-system/oceans/time_series_moorings.html 
* Deployment level metadata and data + data QC flags are provided with each text file. As well as Buoy background and PMEL Carbon program file links. 

**Questions/clarifications for dataset** 
(1) We assumed SST and SSS were IT and SP, respectively, but unsure at time of processing. 
(2) 3-hourly data are provided and were processed and saved. 
Unsure if {SST, SSS, DOXY, CHL, NTU(mostly empty)} were extracted at timestamps 
when 3-hourly MAPCO2 were available, or if averages of {vars} were completed across each 3-hour span.
There are 3-hourly data and then 30 minute data -- not all the data are 3 hourly. 
(*) Can re-run processing if assumptions are off on #1; #2 doesn't matter for data processing steps.

###Obs publication source:
Sutton, A. J., Feely, R. A., Maenner-Jones, S., Musielwicz, S., Osborne, J., Dietrich, C., Monacci, N., Cross, J.,
Bott, R., Kozyr, A., Andersson, A. J., Bates, N. R., Cai, W.-J., Cronin, M. F., De Carlo, E. H., Hales, B., Howden, S. D.,
Lee, C. M., Manzello, D. P., McPhaden, M. J., Meléndez, M., Mickett, J. B., Newton, J. A., Noakes, S. E., Noh, J. H.,
Olafsdottir, S. R., Salisbury, J. E., Send, U., Trull, T. W., Vandemark, D. C., and Weller, R. A. (2019): Autonomous seawater pCO2 and pH time series from 40 surface buoys and the emergence of anthropogenic trends, Earth Syst. Sci. Data, 11, 
421-439, https://doi.org/10.5194/essd-11-421-2019.


