function [Output]=LICANTH(Data)
    % Inputs:
    % Data.Longitude   % X by 1 vector of longitudes (degrees E, positive or negative should be okay)
    % Data.Latitude    % X by 1 vector of latitudes (degrees N)
    % Data.Depth       % X by 1 vector of depths (m)
    % Data.Year        % X by 1 vector of the years of the desired estimates (the exact middle of 2020 would be input as 2020.50)
    % Data.Salinity    % X by 1 vector of salinities (PSS-78)
    % Data.Temperature % X by 1 vector of temperatures (degrees C)
    % Data.Silicate    % X by 1 vector of silicate contents in umol/kg (guess 0 if not sure... never guess a number greater than the Talk)          
    % Data.Phosphate   % X by 1 vector of phosphate contents in umol/kg (guess 0 if not sure... never guess a number greater than the Talk)
    % Data.Talk        % X by 1 vector of total titration seawater alkalinity contents in umol/kg (guess ~2300 if not sure)... used to establish maximum allowable change... use ESPER or LIAR if you'd like a decent estimate from salinity.

    % Output:          Estimates of anthropogenic carbon in the Pacific.
    
    % This code is not well documented here or elsewhere.  S. Please do not
    % share, use at your own peril, and contact Brendan if you are at all
    % unsure whether this code is viable in your region: brendan dot carter
    % at gmail dot com.
    
    % This is where input parsing should go :-(
    
    % Obtaining raw LIR based estimates
    [RawEstimates]=LICANTH_Revisions_NoZ_WideSq(horzcat(Data.Longitude,Data.Latitude,Data.Depth,Data.Year),horzcat(Data.Salinity,Data.Temperature),[1,7],'Equations',[8]);

    % Setting bounds on the estimates based on full equilibration with
    % atmospheric transient... this is needed when the algorithm above is
    % applied in situations that are not spanned by the training data.  It
    % doesn't eliminate all such errors, but it significantly mitigates the
    % impacts of outlier estimates.
    ATMTracer=ReadInAtmTracerRecord();% This was the CO2record upon release
    CO2Rec=load('CO2Trajectories.txt');% Updated to this in 2023_02 
%     ATMTracer.CO2=vertcat([2022 416.45*2-414.24;2021 416.45],ATMTracer.CO2); % appending most recent year estimates
    xCO2=interp1(CO2Rec(:,1),CO2Rec(:,2),Data.Year); % You can change to 2nd index in the 2nd CO2Rec reference to use various SSPs if desired.
    CO2SYSOut280=CO2SYS(Data.Talk,280,1,4,Data.Salinity,Data.Temperature,Data.Temperature,sw_pres(Data.Depth,Data.Latitude),sw_pres(Data.Depth,Data.Latitude),Data.Silicate,Data.Phosphate,1,10,1);
    CO2SYSOutModern=CO2SYS(Data.Talk,xCO2,1,4,Data.Salinity,Data.Temperature,Data.Temperature,sw_pres(Data.Depth,Data.Latitude),sw_pres(Data.Depth,Data.Latitude),Data.Silicate,Data.Phosphate,1,10,1);
    Data.MaxDel=CO2SYSOutModern(:,2)-CO2SYSOut280(:,2);
    Output=min(max(RawEstimates,0),Data.MaxDel);
    Output(isnan(Data.MaxDel))=NaN; % when CO2SYS fails, the seawater composition is such that you should not trust the algorithm estimates
end