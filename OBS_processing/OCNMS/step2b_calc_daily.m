%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This takes prepped data generated by step1_prep_ocnms.m and then
% calculates hourly averages 
% hourly averages for LO import and then also grabs the hourly average at
% 20h00 UTC (12noon local) for model comparisons
% temperature is deg C; oxygen is mg/L
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all

cd /Users/katehewett/Documents/LKH_output/OCNMS_processed/
load('step1_OCNMS_post2011.mat')

A = altitude;
lat = latitude;
lon = longitude;
clear longitude latitude altitude

% check sampling intervals at OCNMS are set to 10ish minutes. check for
% different times in data
ua = unique(diff(timestampsUTC));
if sum(ua ~= minutes(10))>0
    % Shift it to the start of the minute, throwing away seconds and fractional seconds
    startOfMinute = dateshift(timestampsUTC,'start','minute');
    ua = unique(diff(startOfMinute));
    if sum(ua ~= minutes(10))>0
        disp('2x check your sampling timestep')
        return
    end
end

% moving average and display window size
k = (24*60/minutes(ua))+1; 
disp(strcat('sampled @ ='," ",num2str(minutes(ua))," ",'minutes;'," ",'k, hourly window size ='," ",num2str(k)))

clear startOfMinute timestampsLocal mtime
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loop calculates hourly data for S O T P will ave each site as a matfile
% for import to LO obs

% make the time sampling vectors
daily_UTC = dateshift(timestampsUTC(1),'start','day')+hours(20):days(1):dateshift(timestampsUTC(end),'start','day')+hours(20);
%dailyGrab_UTC = dateshift(timestampsUTC(1),'start','day')+hours(20):days(1):dateshift(timestampsUTC(end)+hours(20),'start','day');
pname = '/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs'; 

for sdx = 1:length(site_names)

    sn = site_names{sdx};
    longitude = lon(sdx);
    latitude = lat(sdx);
    Z_est = -depth{sdx};
    altitude = A{sdx};

    timestamp_UTC = daily_UTC'; 

    % calc hourly averages and fill end values with NaN
    tm = movmean(tempC{1,sdx}, k, 1, "omitnan", "EndPoints","fill");
    om = movmean(oxy_mgL{1,sdx}, k, 1, "omitnan", "EndPoints","fill");
    sm = movmean(sal{1,sdx}, k, 1, "omitnan", "EndPoints","fill");
    pm = movmean(pres{1,sdx}, k, 1, "omitnan", "EndPoints","fill");

    figure(sdx)
    TT = tempC{1,sdx};
    plot(timestampsUTC,TT(:,end),'y.'); hold on 
    plot(timestampsUTC,tm(:,end),'c.'); hold on 
    title(sn)

    % interp to match model output hourly times and save
    fn1 = strcat(sn,'_',num2str(min(unique(year(timestampsUTC)))),'_',num2str(max(unique(year(timestampsUTC)))),'_daily.mat');

    IT = interp1(timestampsUTC,tm,timestamp_UTC,'linear');
    OXY = interp1(timestampsUTC,om,timestamp_UTC,'linear');
    SAL = interp1(timestampsUTC,sm,timestamp_UTC,'linear');
    P = interp1(timestampsUTC,pm,timestamp_UTC,'linear');

    plot(timestamp_UTC,IT(:,end),'-k.'); hold on 
    
    cd(pname)
    cd 'daily'
    save(fn1,"sn","longitude","latitude","Z_est","altitude","IT","OXY","SAL","P","timestamp_UTC")

end
