% Read in atmospheric Tracer information
function [ATMTracer]=ReadInAtmTracerRecord()
%     CFC info not needed for this application
    ATMTracer.CO2=load('.\ATM_CO2.txt'); % from here: https://www.ncei.noaa.gov/access/paleo-search/study/10437 supplemented from here after 1959 https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.txt
    ATMTracer.CO2=flipud(ATMTracer.CO2);
end