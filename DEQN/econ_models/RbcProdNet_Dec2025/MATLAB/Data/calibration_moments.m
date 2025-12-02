% The following code was adapted from the replication package of  vom Lehn and Winberry (2019).
% It takes annual industry data from 1948-2017, aggregate and detrend it, and calculate calibration moments.

clc;
clear;

load beadat_37sec;
% Load Consumption data
consdat = xlsread('Real GDP Components.xls',1,'C9:BU9')';
%consdat = xlsread('../Raw Data/Real GDP Components.xls',1,'C9:BU9')';

useHP=0; %if =1, filter data with HP filter; if =0, take log first differences
dim = size(EMP_raw,2);
[yrnum secnum]=size(VA_raw);

%% 1. Aggregate

aggEMP=sum(EMP_raw,2);

% We use Tornqvist index for VA and Investment

% Construct annual shares
EMPsh=EMP_raw./repmat(sum(EMP_raw,2),1,secnum);
VAsh=VAn./repmat(sum(VAn,2),1,secnum);
Domar=1000*GOn./repmat(sum(VAn,2),1,secnum);
Invsh=Invn./repmat(sum(Invn,2),1,secnum);

% Calculate growth rates

aggvagr=zeros(yrnum-1,1);
aggInvgr=zeros(yrnum-1,1);

aggVA=zeros(yrnum,1);
aggInv=zeros(yrnum,1);
aggTFP=zeros(yrnum,1);
aggTFPGO=zeros(yrnum,1);

aggInv(1)=1;
aggVA(1)=1;

% aggVA, aggInv, and aggTFP are constructed via a Tornqvist index
for i=1:yrnum-1
    for j=1:secnum
        aggvagr(i)=aggvagr(i)+(log(VA_raw(i+1,j))-log(VA_raw(i,j)))*0.5*(VAsh(i,j)+VAsh(i+1,j));
        aggInvgr(i)=aggInvgr(i)+(log(InvRaw(i+1,j))-log(InvRaw(i,j)))*0.5*(Invsh(i,j)+Invsh(i+1,j));
    end
    aggVA(i+1)=aggVA(i)*exp(aggvagr(i));
    aggInv(i+1)=aggInv(i)*exp(aggInvgr(i));
end


%% Generate Filtered Data (for business cycle analysis)

if useHP==1
    %%% HP Filter
    for i=1:secnum
        lnEMP(:,i)=log(EMP_raw(:,i))-hpfilter(log(EMP_raw(:,i)),6.25);
        lnVA(:,i)=log(VA_raw(:,i))-hpfilter(log(VA_raw(:,i)),6.25);
        lnInv(:,i)=log(InvRaw(:,i))-hpfilter(log(InvRaw(:,i)),6.25);
    end
    lnVAagg=log(aggVA)-hpfilter(log(aggVA),6.25);
    lnEMPagg=log(aggEMP)-hpfilter(log(aggEMP),6.25);
    lnInvagg=log(aggInv)-hpfilter(log(aggInv),6.25);
    lnCagg=log(consdat)-hpfilter(log(consdat),6.25);
    
    % Omit first three and last three years to avoid endpoint bias
    lnEMP=lnEMP(4:yrnum-3,:);
    lnVA=lnVA(4:yrnum-3,:);
    lnInv=lnInv(4:yrnum-3,:);
    lnTFP=lnTFP(4:yrnum-3,:);
    lnTFPGO=lnTFPGO(4:yrnum-3,:);
    lnEMPagg=lnEMPagg(4:yrnum-3,:);
    lnVAagg=lnVAagg(4:yrnum-3,:);
    lnCagg=lnCagg(4:yrnum-3,:);
    lnInvagg=lnInvagg(4:yrnum-3,:);
    
    % Adjust shares to match time period
    EMPsh=EMPsh(4:yrnum-3,:);
    VAsh=VAsh(4:yrnum-3,:);
    Domar=Domar(4:yrnum-3,:);
    Invsh=Invsh(4:yrnum-3,:);
else
    %%% Growth Rates
    for i=1:secnum
        lnEMP(:,i)=(log(EMP_raw(2:yrnum,i))-log(EMP_raw(1:yrnum-1,i)));
        lnVA(:,i)=(log(VA_raw(2:yrnum,i))-log(VA_raw(1:yrnum-1,i)));
        lnInv(:,i)=(log(InvRaw(2:yrnum,i))-log(InvRaw(1:yrnum-1,i)));

    end
    lnVAagg=log(aggVA(2:yrnum))-log(aggVA(1:yrnum-1));
    lnCagg=log(consdat(2:yrnum))-log(consdat(1:yrnum-1));
    lnEMPagg=log(aggEMP(2:yrnum))-log(aggEMP(1:yrnum-1));
    lnInvagg=log(aggInv(2:yrnum))-log(aggInv(1:yrnum-1));
end


%% 3. Calculate moments

EMPstd=std(lnEMP(:,:));
VAstd=std(lnVA(:,:));
Invstd=std(lnInv(:,:));
EMPaggstd=std(lnEMPagg(:));
VAaggstd=std(lnVAagg(:));
Invaggstd=std(lnInvagg(:));
Caggstd=std(lnCagg(:));
EMPshavg=mean(EMPsh(:,:));
VAshavg=mean(VAsh(:,:));
Invshavg=mean(Invsh(:,:));

save aggregate_moments.mat EMPaggstd VAaggstd Invaggstd Caggstd
