% convert SA CT and DO uM for OCNMS 
% USE GSW so add path 
addpath(genpath('/Users/katehewett/Documents/MATLAB/gsw_matlab_v3_06_16'))

clear all 
close all 


sites = [{'MB015'},{'MB042'},{'CA015'},{'CA042'},{'KL015'},{'KL027'},...
         {'TH015'},{'TH042'},{'CE015'},{'CE042'}];

% hourly 
for sdx = 1:length(sites)

    sn = sites{sdx};
    
    fn = strcat(sn,'_2011_2023_hourly.mat');
    
    cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/hourly/')
    load(fn)
    
    DO_uM = OXY(:,end)*(1000/32); 
    SP = SAL; clear SAL
    
    P = gsw_p_from_z(Z_est,latitude);
    SA = gsw_SA_from_SP(SP, P, longitude, latitude);
    CT = gsw_CT_from_t(SA, IT, P);
    SIG0 = gsw_sigma0(SA,CT);
    
    cd /Users/katehewett/Documents/LKH_output/OCNMS_processed/step4_LOformat/hourly 
    save(fn)
    clearvars -except sdx sites

end

% faux daily  
for sdx = 1:length(sites)

    sn = sites{sdx};
    
    fn = strcat(sn,'_2011_2023_grabdaily.mat');
    
    cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/grab_daily')
    load(fn)
    
    DO_uM = OXY(:,end)*(1000/32); 
    SP = SAL; clear SAL
    
    P = gsw_p_from_z(Z_est,latitude);
    SA = gsw_SA_from_SP(SP, P, longitude, latitude);
    CT = gsw_CT_from_t(SA, IT, P);
    SIG0 = gsw_sigma0(SA,CT);
    
    cd /Users/katehewett/Documents/LKH_output/OCNMS_processed/step4_LOformat/grab_daily 
    save(fn)
    clearvars -except sdx sites

end



