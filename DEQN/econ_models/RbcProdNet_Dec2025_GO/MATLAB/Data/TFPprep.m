% This code loads in measured TFP (constructed separately in Stata), detrends it
% and estimates the autocorrelation of the detrended TFP series.  
clc; clear all;
% load TFPdat_37sec;
TFP = readtable('TFP_37.csv');
TFP_sm = readtable('TFP_37_sm.csv');
TFP_GO = readtable('TFP_GO_37.csv');
TFP_GO_sm = readtable('TFP_GO_37_sm.csv');
TFP_GO_noVA = readtable('TFP_GO_noVA_37.csv');
TFP_GO_noVA_sm = readtable('TFP_GO_noVA_37_sm.csv');

TFP = table2array(TFP(1:end,2:end));
TFP_sm = table2array(TFP_sm(1:end,2:end));
TFP_GO = table2array(TFP_GO(1:end,2:end));
TFP_GO_sm = table2array(TFP_GO_sm(1:end,2:end));
TFP_GO_noVA = table2array(TFP_GO_noVA(1:end,2:end));
TFP_GO_noVA_sm = table2array(TFP_GO_noVA_sm(1:end,2:end));

dim=size(TFP,2);
yrnum = size(TFP,1);
detorder = 4;
%%% Detrend log of TFP (Linear and HP, full sample and pre/post-84)

timetrend=(1:yrnum)';
detTFP = zeros(size(TFP));
detTFP_sm = zeros(size(TFP));
detTFP_GO = zeros(size(TFP));
detTFP_GO_sm = zeros(size(TFP));
detTFP_GO_noVA = zeros(size(TFP));
detTFP_GO_noVA_sm = zeros(size(TFP));

for i=1:dim
    polycoeff = polyfit(timetrend,log(TFP(:,i)),detorder);
    detTFP(:,i) = log(TFP(:,i))-polyval(polycoeff,timetrend);
    polycoeff_sm = polyfit(timetrend,log(TFP_sm(:,i)),detorder);
    detTFP_sm(:,i) = log(TFP_sm(:,i))-polyval(polycoeff_sm,timetrend);
    polycoeff_GO = polyfit(timetrend,log(TFP_GO(:,i)),detorder);
    detTFP_GO(:,i) = log(TFP_GO(:,i))-polyval(polycoeff_GO,timetrend);
    polycoeff_GO_sm = polyfit(timetrend,log(TFP_GO_sm(:,i)),detorder);
    detTFP_GO_sm(:,i) = log(TFP_GO_sm(:,i))-polyval(polycoeff_GO_sm,timetrend);
    polycoeff_GO_noVA = polyfit(timetrend,log(TFP_GO_noVA(:,i)),detorder);
    detTFP_GO_noVA(:,i) = log(TFP_GO_noVA(:,i))-polyval(polycoeff_GO_noVA,timetrend);
    polycoeff_GO_noVA_sm = polyfit(timetrend,log(TFP_GO_noVA_sm(:,i)),detorder);
    detTFP_GO_noVA_sm(:,i) = log(TFP_GO_noVA_sm(:,i))-polyval(polycoeff_GO_noVA_sm,timetrend);
end

%%% Estimate AR(1) coefficients for detrended log TFP in each
%%% sector (MLE)

ar1coeff = zeros(dim,1);
ar1coeff_sm = zeros(dim,1);
ar1coeff_GO = zeros(dim,1);
ar1coeff_GO_sm = zeros(dim,1);
ar1coeff_GO_noVA = zeros(dim,1);
ar1coeff_GO_noVA_sm = zeros(dim,1);

ar1resid = zeros(size(detTFP,1)-1,dim);
ar1resid_sm = zeros(size(detTFP,1)-1,dim);
ar1resid_GO = zeros(size(detTFP,1)-1,dim);
ar1resid_GO_sm = zeros(size(detTFP,1)-1,dim);
ar1resid_GO_noVA = zeros(size(detTFP,1)-1,dim);
ar1resid_GO_noVA_sm = zeros(size(detTFP,1)-1,dim);

for i=1:dim
    ar1coeff(i) = arma_mlear(detTFP(1:yrnum,i),1,0);
    ar1resid(:,i) = detTFP(2:size(detTFP,1),i) - ar1coeff(i)*detTFP(1:size(detTFP,1)-1,i);
    
    ar1coeff_GO(i) = arma_mlear(detTFP_GO(1:yrnum,i),1,0);
    ar1resid_GO(:,i) = detTFP_GO(2:size(detTFP,1),i) - ar1coeff_GO(i)*detTFP_GO(1:size(detTFP,1)-1,i);
    
    ar1coeff_sm(i) = arma_mlear(detTFP_sm(1:yrnum,i),1,0);
    ar1resid_sm(:,i) = detTFP_sm(2:size(detTFP,1),i) - ar1coeff_sm(i)*detTFP(1:size(detTFP_sm,1)-1,i);
    
    ar1coeff_GO_sm(i) = arma_mlear(detTFP_GO_sm(1:yrnum,i),1,0);
    ar1resid_GO_sm(:,i) = detTFP_GO_sm(2:size(detTFP,1),i) - ar1coeff_GO_sm(i)*detTFP_GO_sm(1:size(detTFP,1)-1,i);
    
    ar1coeff_GO_noVA(i) = arma_mlear(detTFP_GO_noVA(1:yrnum,i),1,0);
    ar1resid_GO_noVA(:,i) = detTFP_GO_noVA(2:size(detTFP,1),i) - ar1coeff_GO_noVA(i)*detTFP_GO_noVA(1:size(detTFP,1)-1,i);
    
    ar1coeff_GO_noVA_sm(i) = arma_mlear(detTFP_GO_noVA_sm(1:yrnum,i),1,0);
    ar1resid_GO_noVA_sm(:,i) = detTFP_GO_noVA_sm(2:size(detTFP,1),i) - ar1coeff_GO_noVA_sm(i)*detTFP_GO_noVA_sm(1:size(detTFP,1)-1,i);
end

%%% Construct covariance matrices of TFP, pre/post-84
TFPdatvar = cov(ar1resid);
TFPdatvar_sm = cov(ar1resid_sm);
TFPdatvar_GO = cov(ar1resid_GO);
TFPdatvar_GO_sm = cov(ar1resid_GO_sm);
TFPdatvar_GO_noVA = cov(ar1resid_GO_noVA);
TFPdatvar_GO_noVA_sm = cov(ar1resid_GO_noVA_sm);

modrho = ar1coeff;
modrho_sm = ar1coeff_sm;
modrho_GO = ar1coeff_GO;
modrho_GO_sm = ar1coeff_GO_sm;
modrho_GO_noVA = ar1coeff_GO_noVA;
modrho_GO_noVA_sm = ar1coeff_GO_noVA_sm;

modvcv = TFPdatvar;
modvcv_sm = TFPdatvar_sm;
modvcv_GO = TFPdatvar_GO;
modvcv_GO_sm = TFPdatvar_GO_sm;
modvcv_GO_noVA = TFPdatvar_GO_noVA;
modvcv_GO_noVA_sm = TFPdatvar_GO_noVA_sm;

clearvars -except modrho* modvcv* ar1resid*

save TFP_process;

