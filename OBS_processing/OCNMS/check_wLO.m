
clear all 
close all 

addpath /Users/katehewett/Documents/MATLAB/subaxis
addpath /Users/katehewett/Documents/MATLAB/twentytwo_colors


cd /Users/katehewett/Documents/LO_output/extract/cas6_v0_live/moor/OCNMS_moorings_current
ot = ncread('TH042_2017.01.01_2017.12.31.nc','ocean_time');
lt = datetime(ot, 'convertfrom', 'posixtime', 'Format', 'MM/dd/yy HH:mm:ss.SSS');
lt.TimeZone = 'UTC';

MB042 = ncread('MB042_2017.01.01_2017.12.31.nc','oxygen');
MB042 = MB042(1,:);

CA042 = ncread('CA042_2017.01.01_2017.12.31.nc','oxygen');
CA042 = CA042(1,:);

KL027 = ncread('KL027_2017.01.01_2017.12.31.nc','oxygen');
KL027 = KL027(1,:);

CE042 = ncread('CE042_2017.01.01_2017.12.31.nc','oxygen');
CE042 = CE042(1,:);

TH042 = ncread('TH042_2017.01.01_2017.12.31.nc','oxygen');
TH042 = TH042(1,:);


sn = 'MB042';

subaxis(5,1,1,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);

cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/hourly/')
load(strcat(sn,'_2011_2023_hourly.mat'))

DOobs = OXY(:,end)*(1000/32); 
idx = find(year(timestamp_UTC)~=2017);
DOobs(idx) = []; 
timestamp_UTC(idx) = [];

plot(timestamp_UTC,DOobs,...
    'color',get_matlab_rgb('Red'),'Marker','none',...
    'LineStyle','-','LineWidth',2);
hold on


plot(lt,MB042,...
    'color',[get_matlab_rgb('Navy'), 0.75],'Marker','none',...
    'LineStyle','-','LineWidth',1.5);

ylim([0 300]); 
ax = gca; 
set(gca, 'YTick', [0:50:300])
slabels = string(ax.YAxis.TickLabels); % extract
slabels(2:2:end) = nan; % remove every other one
ax.YAxis.TickLabels = slabels; % set

dlim = [datetime(2017,6,1,0,0,0,0,'TimeZone','UTC') datetime(2017,11,1,0,0,0,0,'TimeZone','UTC')]; 
dtick = [dlim(1):days(15):dlim(2)];
xlim(dlim);
set(gca,'xticklabel',{[]})

set(gca, 'XTick', dtick)
ylabel('DO uM')

fig = gcf;
fig.Color = 'w';

ll1 = legend(sn,'LO','orientation','horizontal','location','best');
legend boxoff 
%%%%%%%%

sn = 'KL027';

subaxis(5,1,4,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);

cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/hourly/')
load(strcat(sn,'_2011_2023_hourly.mat'))

DOobs = OXY(:,end)*(1000/32); 
idx = find(year(timestamp_UTC)~=2017);
DOobs(idx) = []; 
timestamp_UTC(idx) = [];

plot(timestamp_UTC,DOobs,...
    'color',get_matlab_rgb('Red'),'Marker','none',...
    'LineStyle','-','LineWidth',2);
hold on


plot(lt,KL027,...
    'color',[get_matlab_rgb('Navy'), 0.75],'Marker','none',...
    'LineStyle','-','LineWidth',1.5);

ylim([0 300]); 
ax = gca; 
set(gca, 'YTick', [0:50:300])
slabels = string(ax.YAxis.TickLabels); % extract
slabels(2:2:end) = nan; % remove every other one
ax.YAxis.TickLabels = slabels; % set

dlim = [datetime(2017,6,1,0,0,0,0,'TimeZone','UTC') datetime(2017,11,1,0,0,0,0,'TimeZone','UTC')]; 
dtick = [dlim(1):days(15):dlim(2)];
xlim(dlim);
set(gca,'xticklabel',{[]})

set(gca, 'XTick', dtick)
ylabel('DO uM')

fig = gcf;
fig.Color = 'w';

ll1 = legend(sn,'LO','orientation','horizontal','location','best');
legend boxoff 
%%%%%%%%
sn = 'TH042';  
subaxis(5,1,3,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);

cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/hourly/')
load(strcat(sn,'_2011_2023_hourly.mat'))

DOobs = OXY(:,end)*(1000/32); 
idx = find(year(timestamp_UTC)~=2017);
DOobs(idx) = []; 
timestamp_UTC(idx) = [];

plot(timestamp_UTC,DOobs,...
    'color',get_matlab_rgb('Red'),'Marker','none',...
    'LineStyle','-','LineWidth',2);
hold on


plot(lt,TH042,...
    'color',[get_matlab_rgb('Navy'), 0.75],'Marker','none',...
    'LineStyle','-','LineWidth',1.5);

ylim([0 300]); 
ax = gca; 
set(gca, 'YTick', [0:50:300])
slabels = string(ax.YAxis.TickLabels); % extract
slabels(2:2:end) = nan; % remove every other one
ax.YAxis.TickLabels = slabels; % set

dlim = [datetime(2017,6,1,0,0,0,0,'TimeZone','UTC') datetime(2017,11,1,0,0,0,0,'TimeZone','UTC')]; 
dtick = [dlim(1):days(15):dlim(2)];
xlim(dlim);
set(gca, 'XTick', dtick)
ylabel('DO uM')
set(gca,'xticklabel',{[]})

fig = gcf;
fig.Color = 'w';

ll1 = legend(sn,'LO','orientation','horizontal','location','best');
legend boxoff 

%%%%%%%%
sn = 'CE042';  
subaxis(5,1,5,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);

cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/hourly/')
load(strcat(sn,'_2011_2023_hourly.mat'))

DOobs = OXY(:,end)*(1000/32); 
idx = find(year(timestamp_UTC)~=2017);
DOobs(idx) = []; 
timestamp_UTC(idx) = [];

plot(timestamp_UTC,DOobs,...
    'color',get_matlab_rgb('Red'),'Marker','none',...
    'LineStyle','-','LineWidth',2);
hold on


plot(lt,CE042,...
    'color',[get_matlab_rgb('Navy'), 0.75],'Marker','none',...
    'LineStyle','-','LineWidth',1.5);

ylim([0 300]); 
ax = gca; 
set(gca, 'YTick', [0:50:300])
slabels = string(ax.YAxis.TickLabels); % extract
slabels(2:2:end) = nan; % remove every other one
ax.YAxis.TickLabels = slabels; % set

dlim = [datetime(2017,6,1,0,0,0,0,'TimeZone','UTC') datetime(2017,11,1,0,0,0,0,'TimeZone','UTC')]; 
dtick = [dlim(1):days(15):dlim(2)];
xlim(dlim);
set(gca, 'XTick', dtick)
ylabel('DO uM')

fig = gcf;
fig.Color = 'w';

ll1 = legend(sn,'LO','orientation','horizontal','location','best');
legend boxoff 
