% convert SA CT and DO uM for OCNMS
% USE GSW so add path
addpath(genpath('/Users/katehewett/Documents/MATLAB/gsw_matlab_v3_06_16'))

clear all
close all

pYEAR = 2015;

%sites = [{'MB015'},{'MB042'},{'CA015'},{'CA042'},{'KL015'},{'KL027'},...
 %   {'TH015'},{'TH042'},{'CE015'},{'CE042'}];
sites = [{'MB015'},{'CA015'},{'KL015'},{'TH015'},{'CE015'}];

% hourly
for sdx = 1:length(sites)

    sn = sites{sdx};

    fn = strcat(sn,'_2011_2023_hourly.mat');

    cd('/Users/katehewett/Documents/LKH_output/OCNMS_processed/step4_LOformat/hourly')
    load(fn)

    clearvars -except Z_est P CT SP DO_uM sdx sites sn timestamp_UTC pYEAR

    % isolate the year we want to plot
    a = find(year(timestamp_UTC)~=pYEAR);
    CT(a,:) = [];
    SP(a,:) = [];
    DO_uM(a,:) = [];
    timestamp_UTC(a) = [];

    clear a

    %%%%%% SALT
    subaxis(3,1,1,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);

    plot(timestamp_UTC,SP(:,3),...
        'color',get_matlab_rgb('Cyan'),'Marker','none',...
        'LineStyle','-','LineWidth',2);
    hold on

    plot(timestamp_UTC,SP(:,end),...
        'color',get_matlab_rgb('Pink'),'Marker','none',...
        'LineStyle','-','LineWidth',2);
    hold on

    ax1 = gca;

    ylim([24 35]);
    set(ax1, 'YTick', [24:1:35])
    slabels = string(ax1.YAxis.TickLabels); % extract
    slabels(1:2:end) = nan; % remove every other one
    ax1.YAxis.TickLabels = slabels; % set

    dlim = [datetime(pYEAR,6,1,0,0,0,0,'TimeZone','UTC') datetime(pYEAR,11,1,0,0,0,0,'TimeZone','UTC')];
    dtick = [dlim(1):days(15):dlim(2)];
    xlim(dlim);
    set(ax1,'xticklabel',{[]})

    set(ax1, 'XTick', dtick)
    ylabel('SP')

    ax1.YDir = 'reverse'
    set(ax1,'FontSize',11)
    set(ax1,'FontName','Arial')

    fig = gcf;
    fig.Color = 'w';

    %%%%%% T
    subaxis(3,1,2,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);

    plot(timestamp_UTC,CT(:,3),...
        'color',get_matlab_rgb('Cyan'),'Marker','none',...
        'LineStyle','-','LineWidth',2);
    hold on
    plot(timestamp_UTC,CT(:,end),...
        'color',get_matlab_rgb('Pink'),'Marker','none',...
        'LineStyle','-','LineWidth',2);

    ax2 = gca;

    ylim([4 20]);
    set(ax2, 'YTick', [4:2:20])
    slabels = string(ax2.YAxis.TickLabels); % extract
    slabels(2:2:end) = nan; % remove every other one
    ax2.YAxis.TickLabels = slabels; % set

    ax2.YAxisLocation = 'right'

    xlim(dlim);
    set(ax2,'xticklabel',{[]})

    set(ax2, 'XTick', dtick)
    ylabel('CT ^{\circ}C')

    set(ax2,'FontSize',11)
    set(ax2,'FontName','Arial')

    %%%%%% O2
    subaxis(3,1,3,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);

    plot(timestamp_UTC,DO_uM(:,end),...
        'color',get_matlab_rgb('Pink'),'Marker','none',...
        'LineStyle','-','LineWidth',2);
    hold on

    ax3 = gca;

    ylim([0 300]);
    set(ax3, 'YTick', [0:25:300])
    slabels = string(ax3.YAxis.TickLabels); % extract
    slabels(1:2:end) = nan; % remove every other one
    ax3.YAxis.TickLabels = slabels; % set

    xlim(dlim);

    set(ax3, 'XTick', dtick)
    ylabel('DO uM')

    set(ax3,'FontSize',11)
    set(ax3,'FontName','Arial')

    fig = gcf;
    fig.Color = 'w';


    cd /Users/katehewett/Documents/LKH_output/OCNMS_processed/model_comparisons

    cd /Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/moor/OCNMS_moorings_current
    LOfn = strcat(sn,'_',num2str(pYEAR),'.01.01_',num2str(pYEAR),'.12.31.nc');

    ot = ncread(LOfn,'ocean_time');
    lt = datetime(ot, 'convertfrom', 'posixtime', 'Format', 'MM/dd/yy HH:mm:ss.SSS');
    lt.TimeZone = 'UTC';

    LOdo = ncread(LOfn,'oxygen');
    %LOdo = LOdo(2,:);

    LOsalt = ncread(LOfn,'salt');
    LOtemp = ncread(LOfn,'temp');

    LOz = ncread(LOfn,'z_rho');
    
    subaxis(3,1,1,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);
    plot(lt,LOsalt(2,:),...
        'color',[get_matlab_rgb('Maroon'), 0.75],'Marker','none',...
        'LineStyle','-','LineWidth',2);

    plot(lt,LOsalt(22,:),...
        'color',[get_matlab_rgb('Navy'), 0.75],'Marker','none',...
        'LineStyle','-','LineWidth',2);

    ll1 = legend(strcat(sn," (",num2str(Z_est(3)),"m)"),strcat(sn," (",num2str(Z_est(end)),"m)"),...
    'LO', 'LO','orientation','horizontal','location','best');
    legend boxoff

    subaxis(3,1,2,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);
    plot(lt,LOtemp(2,:),...
        'color',[get_matlab_rgb('Maroon'), 0.75],'Marker','none',...
        'LineStyle','-','LineWidth',2);

    plot(lt,LOtemp(22,:),...
        'color',[get_matlab_rgb('Navy'), 0.75],'Marker','none',...
        'LineStyle','-','LineWidth',2);

    subaxis(3,1,3,'Spacing', 0.02, 'Padding', 0 ,'PaddingLeft',0.03,'PaddingRight',0,'Margin', 0.075);
    plot(lt,LOdo(2,:),...
        'color',[get_matlab_rgb('Maroon'), 0.75],'Marker','none',...
        'LineStyle','-','LineWidth',2);

    set(gcf,'position',[131   391   504   324])
     
    cd /Users/katehewett/Documents/LKH_output/OCNMS_processed/model_comparisons
    pname = strcat(LOfn(1:end-3),'.png');
    print(pname,'-dpng','-r750')
   
    ffname = strcat(LOfn(1:end-3),'.fig');
    saveas(gcf,ffname)
    close all 

end
%%





