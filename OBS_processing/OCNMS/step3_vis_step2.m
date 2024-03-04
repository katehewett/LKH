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

sdx = 10; 
sn = site_names{sdx};

plot(timestampsUTC,oxy_mgL{1,sdx},'c.'); hold on

cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/daily')
load(strcat(sn,'_2011_2023_daily.mat'))
plot(timestamp_UTC,OXY(:,end),'-bx'); hold on

cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/grab_daily/')
load(strcat(sn,'_2011_2023_grabdaily.mat'))
plot(timestamp_UTC,OXY(:,end),'-mx'); hold on

return

for sdx = 10 %1:length(site_names)

    cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/daily')

    sn = site_names{sdx};
    load(strcat(sn,'_2011_2023_daily.mat'))

end

return

    longitude = lon(sdx);
    latitude = lat(sdx);
    Z_est = -depth{sdx};
    altitude = A{sdx};

    timestamp_UTC = daily_UTC'; 

    % calc hourly averages and fill end values with NaN
    tm = movmean(tempC{1,sdx}, k, 1, "EndPoints","fill");
    om = movmean(oxy_mgL{1,sdx}, k, 1, "EndPoints","fill");
    sm = movmean(sal{1,sdx}, k, 1, "EndPoints","fill");
    pm = movmean(pres{1,sdx}, k, 1, "EndPoints","fill");

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

