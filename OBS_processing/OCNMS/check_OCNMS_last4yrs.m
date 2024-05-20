% check 2020 - 2023 deployments after email about dubplicates 
% just plotting mooring sites

close all
clear all

cd /Users/katehewett/Documents/LKH_output/OCNMS_processed/step2_sitestructs/hourly

load('TH042_2011_2023_hourly.mat')

yrz = year(timestamp_UTC); 
o = OXY(yrz>2019,:);
s = SAL(yrz>2019,:);
t = IT(yrz>2019,:);
p = P(yrz>2019,:);
tm = timestamp_UTC(yrz>2019);

figure
subplot(2,2,1)
scatter(s(:,end),t(:,end),10,year(tm))
ylabel('T deg C')
xlabel('SP')

xlim([33 34])
ylim([6 10])
title('TH042 TS, color unique 2020-23')


subplot(2,2,[3:4])
plot(tm,o)
ylabel('oxy mgl')
yyaxis right
plot(tm,s)
datetick('x')
ylabel('SP')
title('S DO checks')

set(gcf,'color','w')














