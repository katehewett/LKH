%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step1 of preQC:
% this script loads processed data from Brandy Cervantes google drive, and
% then checks for duplicate dates in the time vector. 
% If there are no dups then it removes dates prior to 2013; 
% Removes sites that are deeper than 42m (b/c sites existed prior to the 
% to the long LO run (2013 start) and current model runs 2017 - present.
% Steps here include renaming 'time' + oxy + temp. And we calc JD + yd
% The results are saved under: 
% cd /Users/katehewett/Documents/LKH_output/OCNMS_processed/...
% step1_OCNMS_post2013.mat
%
% Notes: BC relayed that oxy = mgL and time UTC but not saved in metadata
% 'time' is a matlab builtin --> change to mtime. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all 

cd /Users/katehewett/Documents/LKH_data/ocnms/mooring_data
load('OCNMSMooringData_2000_2023.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 1: rename time + remove dailies, current meters, and deep sites
% original file has time saved as 'time', but that is also a built-in 
% matlab function. reassign name. 
% will not be using current meter data right now
% original file daily centered at 4am local time. Will remove and then 
% re-do averaging so it centers at 20h00 UTC (12noon local). This will
% leave us with 10min original/raw data for the active 10 deployments 

mtime = time; clear time
clear vn ve ve_daily vn_daily dir dir_daily spd spd_daily   
clear oxy_daily pres_daily sal_daily temp_daily time_daily   

oldsites = [5 6 9 12 15];    % just want the 10 active deployments 
site_names(oldsites)=[];  
sal(oldsites)=[];
oxy(oldsites)=[];
pres(oldsites)=[];
temp(oldsites)=[];
depth(oldsites)=[];
altitude(oldsites)=[];
longitude(oldsites)=[];
latitude(oldsites)=[];

clear ans idx oldsites

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 2: Convert timestamps + search for duplicates in the time vector 

timestampsUTC = datetime(mtime,'ConvertFrom','datenum','TimeZone','UTC');
timestampsLocal = datetime(timestampsUTC, 'TimeZone', 'America/Los_Angeles'); 

[uniqueA ia ja] = unique(timestampsUTC,'first');                              
index2Dupesa = find(not(ismember(1:numel(timestampsUTC),ia)));

[uniqueB ib jb] = unique(timestampsLocal,'first');
index2Dupesb = find(not(ismember(1:numel(timestampsLocal),ib)));

if ~isempty(index2Dupesa) | ~isempty(index2Dupesb)
    disp('duplicate timestamps found. Exit.')
    return
elseif isempty(index2Dupesa) & isempty(index2Dupesb)
    disp('no duplicate timestamps found; yay!')
    clear ja jb ia ib index2Dupesa index2Dupesb uniqueA uniqueB
end

% % step 2b: calculate julian dates
% dv = datevec(timestampsUTC);
% jd = juliandate(timestampsUTC);
% 
% [uniqueA i j] = unique(jd,'first');
% index2Dupes = find(not(ismember(1:numel(jd),i)));
% 
% if ~isempty(index2Dupes)
%     disp('duplicates found. Exit.')
%     return
% else
%     disp('no duplicate days found; yay!')
%     clear j i index2Dupes uniqueA
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 3: chop data prior to 2011(2013)
dv = datevec(timestampsUTC);
mYY = dv(:,1);
idx = max(find(mYY<2013));

for sdx = 1:length(site_names)
    figure(sdx)
    subplot(4,1,1); plot(mtime,sal{1,sdx},'rx'); ylabel('sal'); hold on 
    subplot(4,1,2); plot(mtime,oxy{1,sdx},'rx'); ylabel('oxy'); hold on 
    subplot(4,1,3); plot(mtime,temp{1,sdx},'rx'); ylabel('temp'); hold on 
    subplot(4,1,4); plot(mtime,pres{1,sdx},'rx'); ylabel('pres'); hold on 
    title(site_names(sdx));
end

for cdx = 1:10
    sal{1,cdx}(1:idx,:)=[];
    oxy{1,cdx}(1:idx,:)=[];
    pres{1,cdx}(1:idx,:)=[];
    temp{1,cdx}(1:idx,:)=[];
end

mtime(1:idx)=[];
timestampsUTC(1:idx)=[];
timestampsLocal(1:idx)=[];
%dv(1:idx,:)=[];
%jd(1:idx)=[]; 

for sdx = 1:length(site_names)
    figure(sdx)
    subplot(4,1,1); plot(mtime,sal{1,sdx},'c.');
    subplot(4,1,2); plot(mtime,oxy{1,sdx},'c.'); 
    subplot(4,1,3); plot(mtime,temp{1,sdx},'c.'); 
    subplot(4,1,4); plot(mtime,pres{1,sdx},'c.'); 
end


% clean-up and re-name temp and oxy for clarity in next steps
tempC = temp;             
oxy_mgL = oxy; clear oxy

clear sdx idx cdx mYY temp time ans dv

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % step 4: calc julian + year days (Jan 1 = YD 1)
% yrz = unique(year(mtime));     % list of unique years 2013 : 2022 = 11yrs
% yd = ones(1,length(mtime));    % initialize year day vector 
% 
% for ydx = 1:length(yrz)
%     jdi(ydx) = juliandate(yrz(ydx),1,1);
%     grabyear = find(year(mtime)==yrz(ydx));
%     yd(grabyear)=jd(grabyear)-jdi(ydx)+1;
% end
% 
% clear grabyear ans idx jdi ydx

mtime = mtime'; 
timestampsUTC = timestampsUTC'; 
timestampsLocal = timestampsLocal'; 
 
%yd = yd'; 

clear ans

cd /Users/katehewett/Documents/LKH_output/OCNMS_processed
save('step1_OCNMS_post2013')

