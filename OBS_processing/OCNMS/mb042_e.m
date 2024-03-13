% the oxygen data at end of the 2021 deployment at MB042 
% looks suspect. Looking at it here with LO extracted data at that location
% from cas6_v0_live

addpath(genpath('/Users/katehewett/Documents/MATLAB/gsw_matlab_v3_06_16'))

close all
clear all 
cd /Users/katehewett/Documents/LO_output/extract/cas6_v0_live/moor/OCNMS_moorings_current

fn = 'MB042_2018.01.01_2022.12.31.nc';


ot = ncread(fn,'ocean_time');
t = datetime(ot,'ConvertFrom','epochtime');
unique(diff(t))

s = ncread(fn,'salt');
tempi = ncread(fn,'temp');
o = ncread(fn,'oxygen');
zr = ncread(fn,'z_rho');

a = find(year(t)==2021 & month(t)<11 & month(t)>5); % clip to specific season in 2021 
dt = t(a);
salt = s(:,a);
tempc = tempi(:,a);
oxygen = o(:,a);
zrho = zr(:,a);

omed = nanmedian(oxygen,1);
omax = nanmax(oxygen);
omin = nanmin(oxygen);
ob = nanmean(oxygen(1:2,:),1); % last two cells 

% calc o2 sol micro-moles per kg  
% O2sol = gsw_O2sol_SP_pt(SP,pt)
LO_O2sol = gsw_O2sol_SP_pt(salt,tempc);
LO_O2solmed = nanmedian(LO_O2sol,1);
LO_O2solmax = nanmax(LO_O2sol);
LO_O2solmin = nanmin(LO_O2sol);
LO_O2solb = nanmean(LO_O2sol(1:2,:),1); % last two cells 


% some colors 
lyellow = [255 - (255 - [191 146 54])./2]./255;
orange = [186 92 38]./255; 
lblue = [74 135 195]./255;
mblue = [34 65 105]./255;
dblue = [14 30 60]./255;


figure
subplot(2,1,1)
x = datenum(dt)'; y1 = omax; y2 = omin;
plot(x,y1,'color',lyellow); hold on
plot(x,y2,'color',lyellow)
a = patch([x fliplr(x)], [y1 fliplr(y2)], 'g')
a.FaceColor = lyellow;
a.EdgeColor = lyellow;
plot(x,ob,'.','color',orange,'LineStyle','none')

cd /Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/hourly
load('MB042_2011_2023_hourly.mat','OXY','SAL','IT','timestamp_UTC','Z_est')

b = find(year(timestamp_UTC)==2021 & month(timestamp_UTC)<11 & month(timestamp_UTC)>5); % clip to specific season in 2021 
DO = OXY(b,10)*(1000/32);
mt = timestamp_UTC(b); 
SP = SAL(b,10);
it = IT(b,10);

plot(mt,DO,'.','color',mblue,'LineStyle','-')

dt1 = dateshift(datetime(datevec(mt(1))),'start','month') - days(15);
dtn = dateshift(datetime(datevec(mt(end))),'end','month') + days(15);
xlim([datenum(dt1) datenum(dtn)])
ylim([0 600])

% calc o2 sol micro-moles per kg  
% O2sol = gsw_O2sol_SP_pt(SP,pt)
O2sol = gsw_O2sol_SP_pt(SP,it);

subplot(2,1,2)
x = datenum(dt)'; y1 = LO_O2solmax; y2 = LO_O2solmin;
plot(x,y1,'color',lyellow); hold on
plot(x,y2,'color',lyellow)
a = patch([x fliplr(x)], [y1 fliplr(y2)], 'g')
a.FaceColor = lyellow;
a.EdgeColor = lyellow;


plot(mt,O2sol,'.','color','r','LineStyle','-'); hold on 
osat = (DO./O2sol)*100;

yyaxis right 
plot(mt,osat,'.','color','b','LineStyle','-'); hold on 

xlim([datenum(dt1) datenum(dtn)])
ylim([0 600])

set(gcf,'color','white')
return
clear s t o zr t ot 










