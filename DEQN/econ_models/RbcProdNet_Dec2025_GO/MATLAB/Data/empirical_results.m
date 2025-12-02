%%% The following code was adapted from the rep takes annual industry data from 1948-2017 and
%%% the main empirical results in the main text of  vom Lehn and Winberry (2019). 
%%% Christian vom Lehn and Thomas Winberry, 2019.

clc;
clear;

load beadat_37sec;
% load beadat_30sec; % for use in PCA
useHP=0; %if =1, filter data with HP filter; if =0, take log first differences

%%% The data in beatdat_37sec.mat are employment by industry (EMP_raw), real
%%% value added by industry (VA_raw), nominal value added by industry
%%% (VAn), nominal investment expenditures by industry (Invn), real
%%% investment expenditures by industry (InvRaw), nominal gross output by
%%% industry (GOn), TFP constructed using value added (TFP), and TFP
%%% constructed using gross output (TFP_GO).
%%%  These data are drawn from the BEA National Income and Product Accounts, GDP by Industry 
%%% Database, and Fixed Assets Database.  Data have been harmonized to
%%% common industry codes over time (NAICS 1997).  The set of industries
%%% cover the entire non-farm private sector for the United States.
%%% We construct this dataset following procedure described in Appendix A.1.

%%% This dataset also includes shares of total nominal investment production
%%% by industry (Invprodsh), constructed from our investment network 
%%% data (constructed from BEA Input-Output tables), and shares of total 
%%% nominal intermediates production by industry (IOprodsh), also 
%%% constructed from BEA Input-Output tables).

%%% Rows of data correspond to years (1948-2018); columns correspond to
%%% industries.  
%%% In vom Lehn and Winberry (2019), the following sectors are defined as
%%% hub sectors on the basis of investment network data: 
%%% Construction, Machinery, Motor Vehicles, and Professional/
%%% Technical Services.

dim = size(EMP_raw,2);

if dim==37
    hubs = [3 8 11 29];
    nonhubs = [1:2 4:7 9:10 12:28 30:dim];
    mfgnonhubs = [4:7 9:10 12:22];
    intsuppl = [4:7 9:10 12:14 23 25];
    nonhubs_nonintsuppl = [1:2 15:22 24 26:28 30:dim];
    mfgnonhubnonint = (15:22);
elseif dim == 30
    hubs = [3 8 11 22];
    nonhubs = [1:2 4:7 9:10 12:21 23:dim];
    mfgnonhubs = [4:7 9:10 12:15];
    intsuppl = [4:7 9:10 12:14 16 18];
    nonhubs_nonintsuppl = [1:2 15 17 19:21 22:dim];
    mfgnonhubnonint = 15;
end

%%% Structure of codes:
%%% Section 1: construct basic moments to be used in later computations
%%% Section 2: construct all objects necessary to replicate results in main text
%%% Section 3: replicate empirical results from main text


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Constructing Basic Data Moments (aggregates and shares, to be used
% for later computations)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct number of years of data and number of sectors
[yrnum secnum]=size(VA_raw);

% Construct annual shares of total employment, nominal value added, and nominal
% investment (can be done linearly)
EMPsh=EMP_raw./repmat(sum(EMP_raw,2),1,secnum);
VAsh=VAn./repmat(sum(VAn,2),1,secnum);
Domar=1000*GOn./repmat(sum(VAn,2),1,secnum);
Invsh=Invn./repmat(sum(Invn,2),1,secnum);

% Generate Aggregate Real Value Added, Aggregate Real Investment, and Aggregate
% TFP (via a Tornqvist index); Generate Aggregate Employment (via a sum)
aggEMP=sum(EMP_raw,2);

aggvagr=zeros(yrnum-1,1);
aggInvgr=zeros(yrnum-1,1);
aggTFPgr=zeros(yrnum-1,1);
aggTFPGOgr=zeros(yrnum-1,1);

aggVA=zeros(yrnum,1);
aggInv=zeros(yrnum,1);
aggTFP=zeros(yrnum,1);
aggTFPGO=zeros(yrnum,1);

aggInv(1)=1;
aggVA(1)=1;
aggTFP(1)=1;
aggTFPGO(1)=1;

% aggVA, aggInv, and aggTFP are constructed via a Tornqvist index
for i=1:yrnum-1
    for j=1:secnum
        aggvagr(i)=aggvagr(i)+(log(VA_raw(i+1,j))-log(VA_raw(i,j)))*0.5*(VAsh(i,j)+VAsh(i+1,j));
        aggInvgr(i)=aggInvgr(i)+(log(InvRaw(i+1,j))-log(InvRaw(i,j)))*0.5*(Invsh(i,j)+Invsh(i+1,j));
        aggTFPgr(i)=aggTFPgr(i)+(log(TFP(i+1,j))-log(TFP(i,j)))*0.5*(VAsh(i,j)+VAsh(i+1,j));
        aggTFPGOgr(i)=aggTFPGOgr(i)+(log(TFP_GO(i+1,j))-log(TFP_GO(i,j)))*0.5*(Domar(i,j)+Domar(i+1,j));
    end
    aggVA(i+1)=aggVA(i)*exp(aggvagr(i));
    aggInv(i+1)=aggInv(i)*exp(aggInvgr(i));
    aggTFP(i+1)=aggTFP(i)*exp(aggTFPgr(i));
    aggTFPGO(i+1)=aggTFPGO(i)*exp(aggTFPGOgr(i));
end

% Load Consumption data
consdat = xlsread('../Raw Data/Real GDP Components.xls',1,'C9:BU9')';

%% Generate Filtered Data (for business cycle analysis)

if useHP==1
    %%% HP Filter
    for i=1:secnum
        lnEMP(:,i)=log(EMP_raw(:,i))-hpfilter(log(EMP_raw(:,i)),6.25);
        lnVA(:,i)=log(VA_raw(:,i))-hpfilter(log(VA_raw(:,i)),6.25);
        lnInv(:,i)=log(InvRaw(:,i))-hpfilter(log(InvRaw(:,i)),6.25);
        lnTFP(:,i)=log(TFP(:,i))-hpfilter(log(TFP(:,i)),6.25);
        lnTFPGO(:,i)=log(TFP_GO(:,i))-hpfilter(log(TFP_GO(:,i)),6.25);
    end
    lnVAagg=log(aggVA)-hpfilter(log(aggVA),6.25);
    lnEMPagg=log(aggEMP)-hpfilter(log(aggEMP),6.25);
    lnInvagg=log(aggInv)-hpfilter(log(aggInv),6.25);
    lnCagg=log(consdat)-hpfilter(log(consdat),6.25);
    lnTFPagg=log(aggTFP)-hpfilter(log(aggTFP),6.25);
    lnTFPGOagg=log(aggTFPGO)-hpfilter(log(aggTFPGO),6.25);
    
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
    lnTFPagg=lnTFPagg(4:yrnum-3,:);
    lnTFPGOagg=lnTFPGOagg(4:yrnum-3,:);
    
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
        lnTFP(:,i)=(log(TFP(2:yrnum,i))-log(TFP(1:yrnum-1,i)));
        lnTFPGO(:,i)=(log(TFP_GO(2:yrnum,i))-log(TFP_GO(1:yrnum-1,i)));
    end
    lnVAagg=log(aggVA(2:yrnum))-log(aggVA(1:yrnum-1));
    lnCagg=log(consdat(2:yrnum))-log(consdat(1:yrnum-1));
    lnEMPagg=log(aggEMP(2:yrnum))-log(aggEMP(1:yrnum-1));
    lnInvagg=log(aggInv(2:yrnum))-log(aggInv(1:yrnum-1));
    lnTFPagg=log(aggTFP(2:yrnum))-log(aggTFP(1:yrnum-1));
    lnTFPGOagg=log(aggTFPGO(2:yrnum))-log(aggTFPGO(1:yrnum-1));
end

% Define the time windows corresponding to the pre and post 1984 periods.
% Again, first three and last three years of data are omitted to avoid 
% endpoint bias from the filter. 
% 
if useHP==1
    earlywin=(1:33); %1951-1983
    latewin=(34:yrnum-6); %1984-2015
else
    % If doing growth rates, don't need to omit endpoints    
    earlywin = (1:35);
    latewin = (36:yrnum-1);
end

% Define the window length for rolling window analyses
winleng=14;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Generate Variances, Correlations and Avg Shares
% (will be put together into final results tables in the next section of code below)
% (variable names with "2" denote split into pre vs. post 1984 subsamples)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% 
% Variances and shares separated Pre/Post-1984
%%%

% Initialize industry and aggregate variances
EMPvar2=zeros(2,secnum);
TFPvar2=zeros(2,secnum);
TFPGOvar2=zeros(2,secnum);
VAvar2=zeros(2,secnum);
Invvar2=zeros(2,secnum);
EMPvaragg2=zeros(2,1);
VAvaragg2=zeros(2,1);
Invvaragg2=zeros(2,1);
EMPshavg2=zeros(2,secnum);
VAshavg2=zeros(2,secnum);
Invshavg2=zeros(2,secnum);

% Construct variances for industries and aggregate
EMPvar2(1,:)=std(lnEMP(earlywin,:)).^2;
EMPvar2(2,:)=std(lnEMP(latewin,:)).^2;
TFPvar2(1,:)=std(lnTFP(earlywin,:)).^2;
TFPvar2(2,:)=std(lnTFP(latewin,:)).^2;
TFPGOvar2(1,:)=std(lnTFPGO(earlywin,:)).^2;
TFPGOvar2(2,:)=std(lnTFPGO(latewin,:)).^2;
VAvar2(1,:)=std(lnVA(earlywin,:)).^2;
VAvar2(2,:)=std(lnVA(latewin,:)).^2;
Invvar2(1,:)=std(lnInv(earlywin,:)).^2;
Invvar2(2,:)=std(lnInv(latewin,:)).^2;
EMPvaragg2(1)=std(lnEMPagg(earlywin)).^2;
EMPvaragg2(2)=std(lnEMPagg(latewin)).^2;
VAvaragg2(1)=std(lnVAagg(earlywin)).^2;
VAvaragg2(2)=std(lnVAagg(latewin)).^2;
Invvaragg2(1)=std(lnInvagg(earlywin)).^2;
Invvaragg2(2)=std(lnInvagg(latewin)).^2;

% Construct average shares for the pre/post periods
EMPshavg2(1,:)=mean(EMPsh(earlywin,:));
EMPshavg2(2,:)=mean(EMPsh(latewin,:));
VAshavg2(1,:)=mean(VAsh(earlywin,:));
VAshavg2(2,:)=mean(VAsh(latewin,:));
GOshavg2(1,:)=mean(Domar(earlywin,:));
GOshavg2(2,:)=mean(Domar(latewin,:));
Invshavg2(1,:)=mean(Invsh(earlywin,:));
Invshavg2(2,:)=mean(Invsh(latewin,:));

% Construct ratios of employment/GDP variances and Inv/GDP variances
relvolsec2=EMPvar2./VAvar2;
relvolagg2=EMPvaragg2./VAvaragg2;
relvolinvsec2=Invvar2./VAvar2;
relvolinvagg2=Invvaragg2./VAvaragg2;

%%% 
% Variances and shares in Full Sample
% ("t" denotes full sample)
%%%

EMPvart=std(lnEMP(:,:)).^2;
VAvart=std(lnVA(:,:)).^2;
Invvart=std(lnInv(:,:)).^2;
EMPvaraggt=std(lnEMPagg(:)).^2;
VAvaraggt=std(lnVAagg(:)).^2;
Invvaraggt=std(lnInvagg(:)).^2;
EMPshavgt=mean(EMPsh(:,:));
VAshavgt=mean(VAsh(:,:));
Invshavgt=mean(Invsh(:,:));


%%% 
% Generate Correlations: correlation of activity across sectors,  and aggregate/industry 
% correlations between employment and value added and between value added 
% and labor productivity (As above "2" denotes two period, "t" represents whole sample)
%%%


%%% Sector-level Correlations
EMPcorr2=zeros(secnum,secnum,2);
TFPcorr2=zeros(secnum,secnum,2);
TFPGOcorr2=zeros(secnum,secnum,2);
VAcorr2=zeros(secnum,secnum,2);
Invcorr2=zeros(secnum,secnum,2);

EMPcorrt = zeros(secnum,secnum);
VAcorrt = zeros(secnum,secnum);
Invcorrt = zeros(secnum,secnum);

VAEMPindcorr2 = zeros(secnum,2);
VALPindcorr2 = zeros(secnum,2);

for j=1:secnum
    VAEMPindcorr2(j,1)=[1 0]*corrcoef([lnEMP(earlywin,j) lnVA(earlywin,j)])*[0;1];
    VAEMPindcorr2(j,2)=[1 0]*corrcoef([lnEMP(latewin,j) lnVA(latewin,j)])*[0;1];
    VALPindcorr2(j,1) = [1 0]*corrcoef([lnVA(earlywin,j) lnVA(earlywin,j)-lnEMP(earlywin,j)])*[0;1];
    VALPindcorr2(j,2) = [1 0]*corrcoef([lnVA(latewin,j) lnVA(latewin,j)-lnEMP(latewin,j)])*[0;1];
    for k=1:secnum
        EMPcorr2(j,k,1)=[1 0]*corrcoef([lnEMP(earlywin,j) lnEMP(earlywin,k)])*[0;1];
        EMPcorr2(j,k,2)=[1 0]*corrcoef([lnEMP(latewin,j) lnEMP(latewin,k)])*[0;1];
        VAcorr2(j,k,1)=[1 0]*corrcoef([lnVA(earlywin,j) lnVA(earlywin,k)])*[0;1];
        VAcorr2(j,k,2)=[1 0]*corrcoef([lnVA(latewin,j) lnVA(latewin,k)])*[0;1];
        TFPcorr2(j,k,1)=[1 0]*corrcoef([lnTFP(earlywin,j) lnTFP(earlywin,k)])*[0;1];
        TFPcorr2(j,k,2)=[1 0]*corrcoef([lnTFP(latewin,j) lnTFP(latewin,k)])*[0;1];
        TFPGOcorr2(j,k,1)=[1 0]*corrcoef([lnTFPGO(earlywin,j) lnTFPGO(earlywin,k)])*[0;1];
        TFPGOcorr2(j,k,2)=[1 0]*corrcoef([lnTFPGO(latewin,j) lnTFPGO(latewin,k)])*[0;1];
        Invcorr2(j,k,1)=[1 0]*corrcoef([lnInv(earlywin,j) lnInv(earlywin,k)])*[0;1];
        Invcorr2(j,k,2)=[1 0]*corrcoef([lnInv(latewin,j) lnInv(latewin,k)])*[0;1];
        EMPcorrt(j,k)=[1 0]*corrcoef([lnEMP(:,j) lnEMP(:,k)])*[0;1];
        VAcorrt(j,k)=[1 0]*corrcoef([lnVA(:,j) lnVA(:,k)])*[0;1];
        Invcorrt(j,k)=[1 0]*corrcoef([lnInv(:,j) lnInv(:,k)])*[0;1];
    end
end

% Compute weighted pairwise correlations
EMPwcorrfshtot2(1)=sum(sum(((EMPshavgt'*EMPshavgt).*(EMPcorr2(:,:,1)-eye(secnum)))));
EMPwcorrfshtot2(2)=sum(sum(((EMPshavgt'*EMPshavgt).*(EMPcorr2(:,:,2)-eye(secnum)))));
VAwcorrfshtot2(1)=sum(sum(((VAshavgt'*VAshavgt).*(VAcorr2(:,:,1)-eye(secnum)))));
VAwcorrfshtot2(2)=sum(sum(((VAshavgt'*VAshavgt).*(VAcorr2(:,:,2)-eye(secnum)))));

EMPwgtfix=(ones(secnum)-eye(secnum)).*(EMPshavgt'*EMPshavgt);
VAwgtfix=(ones(secnum)-eye(secnum)).*(VAshavgt'*VAshavgt);

EMPwcorrfix2=EMPwcorrfshtot2./sum(sum(EMPwgtfix));
VAwcorrfix2=VAwcorrfshtot2./sum(sum(VAwgtfix));

EMPwcorrtot2=zeros(2,1);
VAwcorrtot2=zeros(2,1);
EMPwcorrtot2(1)=sum(sum(((EMPshavg2(1,:)'*EMPshavgt(1,:)).*(EMPcorr2(:,:,1)-eye(secnum)))));
EMPwcorrtot2(2)=sum(sum(((EMPshavg2(2,:)'*EMPshavg2(2,:)).*(EMPcorr2(:,:,2)-eye(secnum)))));
VAwcorrtot2(1)=sum(sum(((VAshavg2(1,:)'*VAshavg2(1,:)).*(VAcorr2(:,:,1)-eye(secnum)))));
VAwcorrtot2(2)=sum(sum(((VAshavg2(2,:)'*VAshavg2(2,:)).*(VAcorr2(:,:,2)-eye(secnum)))));

EMPwgt2=zeros(2,1);
VAwgt2=zeros(2,1);
EMPwgt2(1)=sum(sum((ones(secnum)-eye(secnum)).*(EMPshavg2(1,:)'*EMPshavg2(1,:))));
EMPwgt2(2)=sum(sum((ones(secnum)-eye(secnum)).*(EMPshavg2(2,:)'*EMPshavg2(2,:))));
VAwgt2(1)=sum(sum((ones(secnum)-eye(secnum)).*(VAshavg2(1,:)'*VAshavg2(1,:))));
VAwgt2(2)=sum(sum((ones(secnum)-eye(secnum)).*(VAshavg2(2,:)'*VAshavg2(2,:))));

EMPwcorr2=EMPwcorrtot2./EMPwgt2;
VAwcorr2=VAwcorrtot2./VAwgt2;

%%% 
% Correlations with aggregate
%%%

% Employment and Value Added
VAEMPcorr2(1)=[1 0]*corrcoef([lnVAagg(earlywin) lnEMPagg(earlywin)])*[0;1];
VAEMPcorr2(2)=[1 0]*corrcoef([lnVAagg(latewin) lnEMPagg(latewin)])*[0;1];

% Value added and Labor Productivity (incl. rolling window)
VALPcorr2(1)=[1 0]*corrcoef([lnVAagg(earlywin) lnVAagg(earlywin)-lnEMPagg(earlywin)])*[0;1];
VALPcorr2(2)=[1 0]*corrcoef([lnVAagg(latewin) lnVAagg(latewin)-lnEMPagg(latewin)])*[0;1];

for i=1:length(lnEMPagg)-(winleng-1)
    VALPcorr(i)=[1 0]*corrcoef([lnVAagg(i:i+(winleng-1)) lnVAagg(i:i+(winleng-1))-lnEMPagg(i:i+(winleng-1))])*[0;1];
end

%%% 
% Correlogram between industry value added and aggregate employment
%%%

correlpre = zeros(secnum,5);
correlpost = zeros(secnum,5);

for i=1:secnum
    correlpre(i,1)= [1 0]*corrcoef([lnVA(1:earlywin(end)-2,i) lnEMPagg(3:earlywin(end))])*[0;1];
    correlpost(i,1)=[1 0]*corrcoef([lnVA(latewin(1):end-2,i) lnEMPagg(latewin(1)+2:end)])*[0;1];
    correlpre(i,2)=[1 0]*corrcoef([lnVA(2:earlywin(end)-1,i) lnEMPagg(3:earlywin(end))])*[0;1];
    correlpost(i,2)=[1 0]*corrcoef([lnVA(latewin(1)+1:end-1,i) lnEMPagg(latewin(1)+2:end)])*[0;1];
    correlpre(i,3)=[1 0]*corrcoef([lnVA(3:earlywin(end),i) lnEMPagg(3:earlywin(end))])*[0;1];
    correlpost(i,3)= [1 0]*corrcoef([lnVA(latewin(1)+2:end,i) lnEMPagg(latewin(1)+2:end)])*[0;1];
    correlpre(i,4)=[1 0]*corrcoef([lnVA(3:earlywin(end),i) lnEMPagg(2:earlywin(end)-1)])*[0;1];
    correlpost(i,4)= [1 0]*corrcoef([lnVA(latewin(1)+2:end,i) lnEMPagg(latewin(1)+1:end-1)])*[0;1];   
    correlpre(i,5)=[1 0]*corrcoef([lnVA(3:earlywin(end),i) lnEMPagg(1:earlywin(end)-2)])*[0;1];
    correlpost(i,5)=[1 0]*corrcoef([lnVA(latewin(1)+2:end,i) lnEMPagg(latewin(1):end-2)])*[0;1];
end

%%% 
% Correlogram between industry value added and aggregate GDP
%%%

correlpre_alt = zeros(secnum,5);
correlpost_alt = zeros(secnum,5);

for i=1:secnum
    correlpre_alt(i,1)= [1 0]*corrcoef([lnVA(1:earlywin(end)-2,i) lnVAagg(3:earlywin(end))])*[0;1];
    correlpost_alt(i,1)=[1 0]*corrcoef([lnVA(latewin(1):end-2,i) lnVAagg(latewin(1)+2:end)])*[0;1];
    correlpre_alt(i,2)=[1 0]*corrcoef([lnVA(2:earlywin(end)-1,i) lnVAagg(3:earlywin(end))])*[0;1];
    correlpost_alt(i,2)=[1 0]*corrcoef([lnVA(latewin(1)+1:end-1,i) lnVAagg(latewin(1)+2:end)])*[0;1];
    correlpre_alt(i,3)=[1 0]*corrcoef([lnVA(3:earlywin(end),i) lnVAagg(3:earlywin(end))])*[0;1];
    correlpost_alt(i,3)= [1 0]*corrcoef([lnVA(latewin(1)+2:end,i) lnVAagg(latewin(1)+2:end)])*[0;1];
    correlpre_alt(i,4)=[1 0]*corrcoef([lnVA(3:earlywin(end),i) lnVAagg(2:earlywin(end)-1)])*[0;1];
    correlpost_alt(i,4)= [1 0]*corrcoef([lnVA(latewin(1)+2:end,i) lnVAagg(latewin(1)+1:end-1)])*[0;1];   
    correlpre_alt(i,5)=[1 0]*corrcoef([lnVA(3:earlywin(end),i) lnVAagg(1:earlywin(end)-2)])*[0;1];
    correlpost_alt(i,5)=[1 0]*corrcoef([lnVA(latewin(1)+2:end,i) lnVAagg(latewin(1):end-2)])*[0;1];
end


%%%
% Generate Variance Decomposition Pieces
%%%

% First generate aggregate variances from decomposition. This is needed
% because the decomposition is approximate.  See paper for details.  As
% above, "2" denotes two period, "t" represents whole sample.

EMPvaraggdec2=zeros(2,1);
VAvaraggdec2=zeros(2,1);
Invvaraggdec2=zeros(2,1);
TFPvaraggdec2=zeros(2,1);
TFPGOvaraggdec2=zeros(2,1);

EMPvaraggdec2(1)=sum(sum(((EMPvar2(1,:).^(1/2))'*((EMPvar2(1,:).^(1/2))).*(EMPshavg2(1,:)'*EMPshavg2(1,:)).*EMPcorr2(:,:,1))));
EMPvaraggdec2(2)=sum(sum(((EMPvar2(2,:).^(1/2))'*((EMPvar2(2,:).^(1/2))).*(EMPshavg2(2,:)'*EMPshavg2(2,:)).*EMPcorr2(:,:,2))));
VAvaraggdec2(1)=sum(sum(((VAvar2(1,:).^(1/2))'*((VAvar2(1,:).^(1/2))).*(VAshavg2(1,:)'*VAshavg2(1,:)).*VAcorr2(:,:,1))));
VAvaraggdec2(2)=sum(sum(((VAvar2(2,:).^(1/2))'*((VAvar2(2,:).^(1/2))).*(VAshavg2(2,:)'*VAshavg2(2,:)).*VAcorr2(:,:,2))));
Invvaraggdec2(1)=sum(sum(((Invvar2(1,:).^(1/2))'*((Invvar2(1,:).^(1/2))).*(Invshavg2(1,:)'*Invshavg2(1,:)).*Invcorr2(:,:,1))));
Invvaraggdec2(2)=sum(sum(((Invvar2(2,:).^(1/2))'*((Invvar2(2,:).^(1/2))).*(Invshavg2(2,:)'*Invshavg2(2,:)).*Invcorr2(:,:,2))));
TFPvaraggdec2(1)=sum(sum(((TFPvar2(1,:).^(1/2))'*((TFPvar2(1,:).^(1/2))).*(VAshavg2(1,:)'*VAshavg2(1,:)).*TFPcorr2(:,:,1))));
TFPvaraggdec2(2)=sum(sum(((TFPvar2(2,:).^(1/2))'*((TFPvar2(2,:).^(1/2))).*(VAshavg2(2,:)'*VAshavg2(2,:)).*TFPcorr2(:,:,2))));
TFPGOvaraggdec2(1)=sum(sum(((TFPGOvar2(1,:).^(1/2))'*((TFPGOvar2(1,:).^(1/2))).*(GOshavg2(1,:)'*GOshavg2(1,:)).*TFPGOcorr2(:,:,1))));
TFPGOvaraggdec2(2)=sum(sum(((TFPGOvar2(2,:).^(1/2))'*((TFPGOvar2(2,:).^(1/2))).*(GOshavg2(2,:)'*GOshavg2(2,:)).*TFPGOcorr2(:,:,2))));

EMPvaraggdec2fsh=zeros(2,1);
VAvaraggdec2fsh=zeros(2,1);

EMPvaraggdec2fsh(1)=sum(sum(((EMPvar2(1,:).^(1/2))'*((EMPvar2(1,:).^(1/2))).*(EMPshavgt'*EMPshavgt).*EMPcorr2(:,:,1))));
EMPvaraggdec2fsh(2)=sum(sum(((EMPvar2(2,:).^(1/2))'*((EMPvar2(2,:).^(1/2))).*(EMPshavgt'*EMPshavgt).*EMPcorr2(:,:,2))));
VAvaraggdec2fsh(1)=sum(sum(((VAvar2(1,:).^(1/2))'*((VAvar2(1,:).^(1/2))).*(VAshavgt'*VAshavgt).*VAcorr2(:,:,1))));
VAvaraggdec2fsh(2)=sum(sum(((VAvar2(2,:).^(1/2))'*((VAvar2(2,:).^(1/2))).*(VAshavgt'*VAshavgt).*VAcorr2(:,:,2))));


% Relative Employment (or Investment) and Value Added Variances from Decomposition
% Aggregates
relvoldec2=EMPvaraggdec2./VAvaraggdec2;
relvoldecfsh2=EMPvaraggdec2fsh./VAvaraggdec2fsh;
relvolinvdec2=Invvaraggdec2./VAvaraggdec2;

% Variances Part of decomposition (denoted with "win" suffix, for within)
EMPvaraggwin2=zeros(2,1);
VAvaraggwin2=zeros(2,1);
Invvaraggwin2=zeros(2,1);
TFPvaraggwin2=zeros(2,1);
TFPGOvaraggwin2=zeros(2,1);

EMPvaraggwin2(1)=sum(sum(((EMPvar2(1,:).^(1/2))'*((EMPvar2(1,:).^(1/2))).*(EMPshavg2(1,:)'*EMPshavg2(1,:)).*eye(secnum))));
EMPvaraggwin2(2)=sum(sum(((EMPvar2(2,:).^(1/2))'*((EMPvar2(2,:).^(1/2))).*(EMPshavg2(2,:)'*EMPshavg2(2,:)).*eye(secnum))));
VAvaraggwin2(1)=sum(sum(((VAvar2(1,:).^(1/2))'*((VAvar2(1,:).^(1/2))).*(VAshavg2(1,:)'*VAshavg2(1,:)).*eye(secnum))));
VAvaraggwin2(2)=sum(sum(((VAvar2(2,:).^(1/2))'*((VAvar2(2,:).^(1/2))).*(VAshavg2(2,:)'*VAshavg2(2,:)).*eye(secnum))));
Invvaraggwin2(1)=sum(sum(((Invvar2(1,:).^(1/2))'*((Invvar2(1,:).^(1/2))).*(Invshavg2(1,:)'*Invshavg2(1,:)).*eye(secnum))));
Invvaraggwin2(2)=sum(sum(((Invvar2(2,:).^(1/2))'*((Invvar2(2,:).^(1/2))).*(Invshavg2(2,:)'*Invshavg2(2,:)).*eye(secnum))));
TFPvaraggwin2(1)=sum(sum(((TFPvar2(1,:).^(1/2))'*((TFPvar2(1,:).^(1/2))).*(VAshavg2(1,:)'*VAshavg2(1,:)).*eye(secnum))));
TFPvaraggwin2(2)=sum(sum(((TFPvar2(2,:).^(1/2))'*((TFPvar2(2,:).^(1/2))).*(VAshavg2(2,:)'*VAshavg2(2,:)).*eye(secnum))));
TFPGOvaraggwin2(1)=sum(sum(((TFPGOvar2(1,:).^(1/2))'*((TFPGOvar2(1,:).^(1/2))).*(GOshavg2(1,:)'*GOshavg2(1,:)).*eye(secnum))));
TFPGOvaraggwin2(2)=sum(sum(((TFPGOvar2(2,:).^(1/2))'*((TFPGOvar2(2,:).^(1/2))).*(GOshavg2(2,:)'*GOshavg2(2,:)).*eye(secnum))));

EMPvaraggwin2fsh=zeros(2,1);
VAvaraggwin2fsh=zeros(2,1);

EMPvaraggwin2fsh(1)=sum(sum(((EMPvar2(1,:).^(1/2))'*((EMPvar2(1,:).^(1/2))).*(EMPshavgt'*EMPshavgt).*eye(secnum))));
EMPvaraggwin2fsh(2)=sum(sum(((EMPvar2(2,:).^(1/2))'*((EMPvar2(2,:).^(1/2))).*(EMPshavgt'*EMPshavgt).*eye(secnum))));
VAvaraggwin2fsh(1)=sum(sum(((VAvar2(1,:).^(1/2))'*((VAvar2(1,:).^(1/2))).*(VAshavgt'*VAshavgt).*eye(secnum))));
VAvaraggwin2fsh(2)=sum(sum(((VAvar2(2,:).^(1/2))'*((VAvar2(2,:).^(1/2))).*(VAshavgt'*VAshavgt).*eye(secnum))));

relvolwin2=EMPvaraggwin2./VAvaraggwin2;
relvolwinfsh2=EMPvaraggwin2fsh./VAvaraggwin2fsh;
relvolinvwin2=Invvaraggwin2./VAvaraggwin2;

% Covariances Part of decomposition (denoted with "bet" suffix, for between)

EMPvaraggbet2=zeros(2,1);
VAvaraggbet2=zeros(2,1);
Invvaraggbet2=zeros(2,1);
TFPvaraggbet2=zeros(2,1);
TFPGOvaraggbet2=zeros(2,1);

EMPvaraggbet2(1)=sum(sum(((EMPvar2(1,:).^(1/2))'*((EMPvar2(1,:).^(1/2))).*(EMPshavg2(1,:)'*EMPshavg2(1,:)).*(EMPcorr2(:,:,1).*(ones(secnum)-eye(secnum))))));
EMPvaraggbet2(2)=sum(sum(((EMPvar2(2,:).^(1/2))'*((EMPvar2(2,:).^(1/2))).*(EMPshavg2(2,:)'*EMPshavg2(2,:)).*(EMPcorr2(:,:,2).*(ones(secnum)-eye(secnum))))));
VAvaraggbet2(1)=sum(sum(((VAvar2(1,:).^(1/2))'*((VAvar2(1,:).^(1/2))).*(VAshavg2(1,:)'*VAshavg2(1,:)).*(VAcorr2(:,:,1).*(ones(secnum)-eye(secnum))))));
VAvaraggbet2(2)=sum(sum(((VAvar2(2,:).^(1/2))'*((VAvar2(2,:).^(1/2))).*(VAshavg2(2,:)'*VAshavg2(2,:)).*(VAcorr2(:,:,2).*(ones(secnum)-eye(secnum))))));
Invvaraggbet2(1)=sum(sum(((Invvar2(1,:).^(1/2))'*((Invvar2(1,:).^(1/2))).*(Invshavg2(1,:)'*Invshavg2(1,:)).*(Invcorr2(:,:,1).*(ones(secnum)-eye(secnum))))));
Invvaraggbet2(2)=sum(sum(((Invvar2(2,:).^(1/2))'*((Invvar2(2,:).^(1/2))).*(Invshavg2(2,:)'*Invshavg2(2,:)).*(Invcorr2(:,:,2).*(ones(secnum)-eye(secnum))))));
TFPvaraggbet2(1)=sum(sum(((TFPvar2(1,:).^(1/2))'*((TFPvar2(1,:).^(1/2))).*(VAshavg2(1,:)'*VAshavg2(1,:)).*(TFPcorr2(:,:,1).*(ones(secnum)-eye(secnum))))));
TFPvaraggbet2(2)=sum(sum(((TFPvar2(2,:).^(1/2))'*((TFPvar2(2,:).^(1/2))).*(VAshavg2(2,:)'*VAshavg2(2,:)).*(TFPcorr2(:,:,2).*(ones(secnum)-eye(secnum))))));
TFPGOvaraggbet2(1)=sum(sum(((TFPGOvar2(1,:).^(1/2))'*((TFPGOvar2(1,:).^(1/2))).*(GOshavg2(1,:)'*GOshavg2(1,:)).*(TFPGOcorr2(:,:,1).*(ones(secnum)-eye(secnum))))));
TFPGOvaraggbet2(2)=sum(sum(((TFPGOvar2(2,:).^(1/2))'*((TFPGOvar2(2,:).^(1/2))).*(GOshavg2(2,:)'*GOshavg2(2,:)).*(TFPGOcorr2(:,:,2).*(ones(secnum)-eye(secnum))))));

EMPvaraggbet2fsh=zeros(2,1);
VAvaraggbet2fsh=zeros(2,1);

EMPvaraggbet2fsh(1)=sum(sum(((EMPvar2(1,:).^(1/2))'*((EMPvar2(1,:).^(1/2))).*(EMPshavgt'*EMPshavgt).*(EMPcorr2(:,:,1).*(ones(secnum)-eye(secnum))))));
EMPvaraggbet2fsh(2)=sum(sum(((EMPvar2(2,:).^(1/2))'*((EMPvar2(2,:).^(1/2))).*(EMPshavgt'*EMPshavgt).*(EMPcorr2(:,:,2).*(ones(secnum)-eye(secnum))))));
VAvaraggbet2fsh(1)=sum(sum(((VAvar2(1,:).^(1/2))'*((VAvar2(1,:).^(1/2))).*(VAshavgt'*VAshavgt).*(VAcorr2(:,:,1).*(ones(secnum)-eye(secnum))))));
VAvaraggbet2fsh(2)=sum(sum(((VAvar2(2,:).^(1/2))'*((VAvar2(2,:).^(1/2))).*(VAshavgt'*VAshavgt).*(VAcorr2(:,:,2).*(ones(secnum)-eye(secnum))))));


relvolbet2=EMPvaraggbet2./VAvaraggbet2;
relvolbetfsh2=EMPvaraggbet2fsh./VAvaraggbet2fsh;
relvolinvbet2=Invvaraggbet2./VAvaraggbet2;

%Variance Weights%
VAwinwgt2=VAvaraggwin2./VAvaraggdec2;
VAwinwgtfsh2=VAvaraggwin2fsh./VAvaraggdec2fsh;

%Within vs. Between Contributions (multiplied by weights)%%

relvolwincon2=relvolwin2.*VAwinwgt2;
relvolbetcon2=relvolbet2.*(1-VAwinwgt2);
relvolinvwincon2=relvolinvwin2.*VAwinwgt2;
relvolinvbetcon2=relvolinvbet2.*(1-VAwinwgt2);

wincon = (relvolwincon2(2)-relvolwincon2(1))./(relvolwincon2(2)+relvolbetcon2(2)-relvolwincon2(1)-relvolbetcon2(1));
betwcon = (relvolbetcon2(2)-relvolbetcon2(1))./(relvolwincon2(2)+relvolbetcon2(2)-relvolwincon2(1)-relvolbetcon2(1));

winconinv = (relvolinvwincon2(2)-relvolinvwincon2(1))./(relvolinvwincon2(2)+relvolinvbetcon2(2)-relvolinvwincon2(1)-relvolinvbetcon2(1));
betwconinv = (relvolinvbetcon2(2)-relvolinvbetcon2(1))./(relvolinvwincon2(2)+relvolinvbetcon2(2)-relvolinvwincon2(1)-relvolinvbetcon2(1));

relvolwinconfsh2=relvolwinfsh2.*VAwinwgtfsh2;
relvolbetconfsh2=relvolbetfsh2.*(1-VAwinwgtfsh2);

winconfsh = (relvolwinconfsh2(2)-relvolwinconfsh2(1))./(relvolwinconfsh2(2)+relvolbetconfsh2(2)-relvolwinconfsh2(1)-relvolbetconfsh2(1));
betwconfsh = (relvolbetconfsh2(2)-relvolbetconfsh2(1))./(relvolwinconfsh2(2)+relvolbetconfsh2(2)-relvolwinconfsh2(1)-relvolbetconfsh2(1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Results in main text
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
% Section 2 Results
%%%

%%% Not in a table/figure in the paper, but the following computes the shares of economic activty by hubs and
%%% nonhubs for outcomes like employment, GDP, investment, investment
%%% produced, etc.
% VAshfull=VAn./repmat(sum(VAn,2),1,secnum); % Since existing VA share object has had endpoints removed to match filter
% EMPshfull=EMP_raw./repmat(sum(EMP_raw,2),1,secnum); % Since existing VA share object has had endpoints removed to match filter
% Invshfull=Invn./repmat(sum(Invn,2),1,secnum); % Since existing VA share object has had endpoints removed to match filter
% % Investment produced
% table2a = [mean(sum(Invprodsh(hubs,2:37),1)) mean(sum(Invprodsh(hubs,38:72),1)) 1-mean(sum(Invprodsh(hubs,2:37),1)) 1-mean(sum(Invprodsh(hubs,38:72),1))];
% % Value added produced
% table2b = [mean(sum(VAshfull(1:36,hubs),2)) mean(sum(VAshfull(37:71,hubs),2)) 1-mean(sum(VAshfull(1:36,hubs),2)) 1-mean(sum(VAshfull(37:71,hubs),2))];
% % Intermediates produced
% table2c = [mean(sum(IOprodsh(hubs,2:37),1)) mean(sum(IOprodsh(hubs,38:72),1)) 1-mean(sum(IOprodsh(hubs,2:37),1)) 1-mean(sum(IOprodsh(hubs,38:72),1))];
% % Employment
% table2d = [mean(sum(EMPshfull(1:36,hubs),2)) mean(sum(EMPshfull(37:71,hubs),2)) 1-mean(sum(EMPshfull(1:36,hubs),2)) 1-mean(sum(EMPshfull(37:71,hubs),2))];
% % Investment purchased
% table2e = [mean(sum(Invshfull(1:36,hubs),2)) mean(sum(Invshfull(37:71,hubs),2)) 1-mean(sum(Invshfull(1:36,hubs),2)) 1-mean(sum(Invshfull(37:71,hubs),2))];
% 
% table2 = [table2a;table2b;table2c;table2d;table2e];


%%% Table 2
% Value Added Variances
table2a = 100*[mean(sqrt(VAvar2(1,hubs))) mean(sqrt(VAvar2(2,hubs))) mean(sqrt(VAvar2(1,nonhubs))) mean(sqrt(VAvar2(2,nonhubs)))];
% Employment Variances
table2b = 100*[mean(sqrt(EMPvar2(1,hubs))) mean(sqrt(EMPvar2(2,hubs))) mean(sqrt(EMPvar2(1,nonhubs))) mean(sqrt(EMPvar2(2,nonhubs)))];
table2 = [table2a;table2b];

%%% Figure 2

% Pre-1984 data for figure
figure2a = [mean(correlpre(hubs,:))' mean(correlpost(hubs,:))'];
% Post-1984 data for figure
figure2b = [mean(correlpre(nonhubs,:))' mean(correlpost(nonhubs,:))'];

figure2 = [figure2a figure2b];

%%%
% Section 5 Results
%%%

%%% Table 3

% Aggregate Variance
table3a = 1000*[TFPGOvaraggdec2(1) TFPGOvaraggdec2(2) VAvaraggdec2(1) VAvaraggdec2(2)];
table3b = 1000*[TFPGOvaraggwin2(1) TFPGOvaraggwin2(2) VAvaraggwin2(1) VAvaraggwin2(2)];
table3c = 1000*[TFPGOvaraggbet2(1) TFPGOvaraggbet2(2) VAvaraggbet2(1) VAvaraggbet2(2)];

table3 = [table3a;table3b;table3c];

%%% Table 4 (Data only)
dinvsh=Invsh(2:end,:)-Invsh(1:end-1,:);
table4 = 1000*[mean(mean(abs(dinvsh)));mean(std(dinvsh))];

%%% Table 5 (Data only)

% GDP variance
table5a = 100*[sqrt(VAvaraggdec2(1)) sqrt(VAvaraggdec2(2))];
% GDP-Productivity Correlation
table5b = [VALPcorr2(1) VALPcorr2(2)];
% Relative Std dev of Employment to Output
table5c = [sqrt(relvolagg2(1)) sqrt(relvolagg2(2))];
% Relative Std dev of Investment Expenditures to Output
table5d = [sqrt(Invvaragg2./VAvaragg2)'];

table5 = [table5a;table5b;table5c;table5d];


%%% Figure 7 (Data content only)
figure10 = VALPcorr';

%%%
% Section 6 Results
%%%

%%% Table 8 (and H.2 if HP filter on) (Data only)

% GDP variance
table8a = 100*[sum(VAshavg2.*sqrt(VAvar2),2)'];
% GDP-Productivity Correlation
table8b = [sum(VAshavg2.*(VALPindcorr2'),2)'];
% Relative Std dev of Employment to Output
table8c = [sum(VAshavg2.*sqrt(relvolsec2),2)'];
table8 = [table8a;table8b;table8c];

%%% Table 9 (and H.8 if HP filter on) (Data only)

% Relative Employment Variance
table9a = [relvoldec2' 100];
% Variance Component
table9b = [relvolwin2' 100*wincon];
% Covariance Component
table9c = [relvolbet2' 100*betwcon];
% Variance Weight
table9d = [VAwinwgt2'];
table9 = [table9a;table9b;table9c;[table9d 100]];

%%% 
% Appendix Results
%%% 

%%% Table B.1
% Value Added Variances
tableb1a = 100*[mean(sqrt(VAvar2(1,hubs))) mean(sqrt(VAvar2(2,hubs))) mean(sqrt(VAvar2(1,nonhubs))) mean(sqrt(VAvar2(2,nonhubs))) mean(sqrt(VAvar2(1,mfgnonhubs))) mean(sqrt(VAvar2(2,mfgnonhubs)))];
% Employment Variances
tableb1b = 100*[mean(sqrt(EMPvar2(1,hubs))) mean(sqrt(EMPvar2(2,hubs))) mean(sqrt(EMPvar2(1,nonhubs))) mean(sqrt(EMPvar2(2,nonhubs))) mean(sqrt(EMPvar2(1,mfgnonhubs))) mean(sqrt(EMPvar2(2,mfgnonhubs)))];
tableb1 = [tableb1a;tableb1b];


%%% Figure B.1

% Pre-1984 data for figure
figureb1a = [mean(correlpre_alt(hubs,:))' mean(correlpost_alt(hubs,:))'];
% Post-1984 data for figure
figureb1b = [mean(correlpre_alt(nonhubs,:))' mean(correlpost_alt(nonhubs,:))'];

figureb1 = [figureb1a figureb1b];

%%% Figure B.2

% Hubs data for figure
figureb2a = [mean(correlpre(hubs,:))' mean(correlpost(hubs,:))'];
% Non-Hubs data for figure
figureb2b = [mean(correlpre(nonhubs,:))' mean(correlpost(nonhubs,:))'];
% Mfg Non-Hubs data for figure
figureb2c = [mean(correlpre(mfgnonhubs,:))' mean(correlpost(mfgnonhubs,:))'];

figureb2 = [figureb2a figureb2b figureb2c];

%%% Table D.1
% NOTE: Only meaningful when using 30 sector data!

cut=35; % 1983, in growth rates
tabled1_pcatable_TFPGO = pcaprepost(TFP_GO,log(aggTFPGO(2:length(aggVA))./aggTFPGO(1:length(aggVA)-1)),cut,1);


%%% Table F.1
if dim==37
    conshubs = [15 24 27:28 33 36:37];
    tablef1 = [mean(mean(EMPcorr2(hubs,conshubs,1))) mean(mean(EMPcorr2(hubs,conshubs,2)))];
end

%%% Table F.2
% Value Added Variances
tablef2a = 100*[mean(sqrt(VAvar2(1,hubs))) mean(sqrt(VAvar2(2,hubs))) mean(sqrt(VAvar2(1,intsuppl))) mean(sqrt(VAvar2(2,intsuppl))) mean(sqrt(VAvar2(1,nonhubs_nonintsuppl))) mean(sqrt(VAvar2(2,nonhubs_nonintsuppl)))];
% Employment Variances
tablef2b = 100*[mean(sqrt(EMPvar2(1,hubs))) mean(sqrt(EMPvar2(2,hubs))) mean(sqrt(EMPvar2(1,intsuppl))) mean(sqrt(EMPvar2(2,intsuppl))) mean(sqrt(EMPvar2(1,nonhubs_nonintsuppl))) mean(sqrt(EMPvar2(2,nonhubs_nonintsuppl)))];
tablef2 = [tablef2a;tablef2b];

%%% Figure F.2
% Hubs
figuref2a = [mean(correlpre(hubs,:))' mean(correlpost(hubs,:))'];
% Intermediate Suppliers
figuref2b = [mean(correlpre(intsuppl,:))' mean(correlpost(intsuppl,:))'];
% Non Hubs, Non-Intermediate Suppliers
figuref2c = [mean(correlpre(nonhubs_nonintsuppl,:))' mean(correlpost(nonhubs_nonintsuppl,:))'];

figuref2 = [figuref2a figuref2b figuref2c];

%%% Figure G.1 and G.2 (depending on filter settings) (Data only)

figureg1 = [lnVAagg lnEMPagg lnInvagg];

%%% Table H.1
VALPcorr2fixcorr = VALPcorr2;
VALPcorr2fixcorr(2) = (1-sqrt(relvolagg2(2))*VAEMPcorr2(1))/sqrt(1+relvolagg2(2)-2*sqrt(relvolagg2(2))*VAEMPcorr2(1));
VALPcorr2fixsd = VALPcorr2;
VALPcorr2fixsd(2) = (1-sqrt(relvolagg2(1))*VAEMPcorr2(2))/sqrt(1+relvolagg2(1)-2*sqrt(relvolagg2(1))*VAEMPcorr2(2));

tableh1 = [VALPcorr2;VAEMPcorr2;VALPcorr2fixsd;sqrt(relvolagg2');VALPcorr2fixcorr];

%%% Table H.3

% Fixed Weights
% GDP variance
tableh3a_fw = 100*[sum(repmat(VAshavgt,2,1).*sqrt(VAvar2),2)'];
% GDP-Productivity Correlation
tableh3b_fw = [sum(repmat(VAshavgt,2,1).*(VALPindcorr2'),2)'];
% Relative Std dev of Employment to Output
tableh3c_fw = [sum(repmat(VAshavgt,2,1).*sqrt(relvolsec2),2)'];
tableh3_fw = [tableh3a_fw;tableh3b_fw;tableh3c_fw];

% No Weights
% GDP variance
tableh3a_nw = 100*[sum(1/dim.*sqrt(VAvar2),2)'];
% GDP-Productivity Correlation
tableh3b_nw = [sum(1/dim.*(VALPindcorr2'),2)'];
% Relative Std dev of Employment to Output
tableh3c_nw = [sum(1/dim.*sqrt(relvolsec2),2)'];
tableh3_nw = [tableh3a_nw;tableh3b_nw;tableh3c_nw];

%%% Table H.4
tableh4 = [relvolagg2';relvoldec2';sqrt(relvolagg2');sqrt(relvoldec2)'];

%%% Table H.5 (Data only)

% Pre-84 Pairwise Correlations
tableh5a = [EMPwcorr2(1) VAwcorr2(1)]; 
% Post-84 Pairwise Correlations
tableh5b = [EMPwcorr2(2) VAwcorr2(2)];
tableh5 = [tableh5a;tableh5b];

%%% Figure H.1 (see below)

%%% Table H.7
% Relative Employment Variance
tableh7a = [relvoldecfsh2' 100];
% Variance Component
tableh7b = [relvolwinfsh2' 100*winconfsh];
% Covariance Component
tableh7c = [relvolbetfsh2' 100*betwconfsh];
% Variance Weight
tableh7d = [VAwinwgtfsh2'];
tableh7 = [tableh7a;tableh7b;tableh7c;[tableh7d 100]];

%%% Misc. results regarding relative volatility at hubs vs. nonhubs

table_misc_a = [mean(sqrt(EMPvar2(:,hubs)),2)./mean(sqrt(VAvar2(:,hubs)),2) mean(sqrt(EMPvar2(:,nonhubs)),2)./mean(sqrt(VAvar2(:,nonhubs)),2)];
table_misc_b = [(sum(EMPshavg2(:,hubs).*sqrt(EMPvar2(:,hubs)),2)./sum(EMPshavg2(:,hubs),2))./(sum(VAshavg2(:,hubs).*sqrt(VAvar2(:,hubs)),2)./sum(VAshavg2(:,hubs),2)) (sum(EMPshavg2(:,nonhubs).*sqrt(EMPvar2(:,nonhubs)),2)./sum(EMPshavg2(:,nonhubs),2))./(sum(VAshavg2(:,nonhubs).*sqrt(VAvar2(:,nonhubs)),2)./sum(VAshavg2(:,nonhubs),2))];
table_misc_c = [(sum(EMPshavgt(:,hubs).*sqrt(EMPvar2(:,hubs)),2)./sum(EMPshavgt(:,hubs),2))./(sum(VAshavgt(:,hubs).*sqrt(VAvar2(:,hubs)),2)./sum(VAshavgt(:,hubs),2)) (sum(EMPshavgt(:,nonhubs).*sqrt(EMPvar2(:,nonhubs)),2)./sum(EMPshavgt(:,nonhubs),2))./(sum(VAshavgt(:,nonhubs).*sqrt(VAvar2(:,nonhubs)),2)./sum(VAshavg2(:,nonhubs),2))];
table_misc_d = [mean(sqrt(relvolsec2(:,hubs)),2) mean(sqrt(relvolsec2(:,nonhubs)),2)];
table_misc_e = [sum(VAshavg2(:,hubs).*sqrt(relvolsec2(:,hubs)),2)./sum(VAshavg2(:,hubs),2) sum(VAshavg2(:,nonhubs).*sqrt(relvolsec2(:,nonhubs)),2)./sum(VAshavg2(:,nonhubs),2)];
table_misc_f = [sum(VAshavgt(:,hubs).*sqrt(relvolsec2(:,hubs)),2)./sum(VAshavgt(:,hubs),2) sum(VAshavgt(:,nonhubs).*sqrt(relvolsec2(:,nonhubs)),2)./sum(VAshavgt(:,nonhubs),2)];
table_misc = [table_misc_a;table_misc_b;table_misc_c;table_misc_d;table_misc_e;table_misc_f];

%%% Plotting detrended TFP (Figure D.5)

% Detrend construction and machinery manufacturing sectors at different
% levels 

Const_TFP = log(TFP_GO(:,3));
Mach_TFP = log(TFP_GO(:,8));
timetrend=(1:yrnum)';

Const_TFP_1 = polyval(polyfit(timetrend,Const_TFP,1),timetrend);
Const_TFP_2 = polyval(polyfit(timetrend,Const_TFP,2),timetrend);
Const_TFP_4 = polyval(polyfit(timetrend,Const_TFP,4),timetrend);

Mach_TFP_1 = polyval(polyfit(timetrend,Mach_TFP,1),timetrend);
Mach_TFP_2 = polyval(polyfit(timetrend,Mach_TFP,2),timetrend);
Mach_TFP_4 = polyval(polyfit(timetrend,Mach_TFP,4),timetrend);

vConstruction_0 = Const_TFP;
vConstruction_1 = Const_TFP_1;
vConstruction_2 = Const_TFP_2;
vConstruction_4 = Const_TFP_4;

vMachinery_0 = Mach_TFP;
vMachinery_1 = Mach_TFP_1;
vMachinery_2 = Mach_TFP_2;
vMachinery_4 = Mach_TFP_4;

[T,~] = size(vConstruction_0);

startDate   = datenum('01-01-1948');
endDate     = datenum('01-01-2018');
xData       = linspace(startDate,endDate,T)';

bblue     = [8/255,62/255,118/255];
rred      = [178/255,34/255,34/255];
ppurp     = [.5,.24,.5];

figure

h               = gcf;
h.PaperUnits    = 'inches';
h.PaperPosition = [0 0 9 4];

subplot(1,2,1)
hold on
plot(xData,vConstruction_0,'linewidth',1.5,'linestyle','-','color','k')
plot(xData,vConstruction_1,'linewidth',1.5,'linestyle','--','color',rred)
plot(xData,vConstruction_2,'linewidth',1.5,'linestyle','-.','color',ppurp)
plot(xData,vConstruction_4,'linewidth',1.5,'linestyle',':','color',bblue)
xlabel('Years','interpreter','latex')
ylabel('Log TFP','interpreter','latex')
title('Construction','interpreter','latex','fontsize',16)
datetick('x','yyyy','keeplimits')
xlim([startDate endDate])
set(gcf,'color','w')
grid on
hold off

subplot(1,2,2)
hold on
plot(xData,vMachinery_0,'linewidth',1.5,'linestyle','-','color','k')
plot(xData,vMachinery_1,'linewidth',1.5,'linestyle','--','color',rred)
plot(xData,vMachinery_2,'linewidth',1.5,'linestyle','-.','color',ppurp)
plot(xData,vMachinery_4,'linewidth',1.5,'linestyle',':','color',bblue)
xlabel('Years','interpreter','latex')
ylabel('Log TFP','interpreter','latex')
ylim([-.1 .5])
title('Machinery Manufacturing','interpreter','latex','fontsize',16)
datetick('x','yyyy','keeplimits')
xlim([startDate endDate])
h	 = legend('Data','First order','Second order','Fourth order');
set(h,'interpreter','latex','location','southeast','fontsize',12)
set(gcf,'color','w')
grid on
hold off

print('detrending.eps','-depsc')


%%% Covariance Figures
% Construct weights for marker size
mVAWeight       = VAshavgt' * VAshavgt;

Z = reshape(nonzeros(triu(mVAWeight,1)),[],1);
Z = 100*(Z / mean(Z));

%%% 
% Produce Correlograms
%%%
if useHP==0 && dim==37
    produce_correlogram;
end

%%%
% Plot of covariances against each other (Figure H.1)
%%%

X = (reshape(nonzeros(triu(cov(lnVA(latewin,:)),1)),[],1) - reshape(nonzeros(triu(cov(lnVA(earlywin,:)),1)),[],1));
Y =(reshape(nonzeros(triu(cov(lnEMP(latewin,:)),1)),[],1) - reshape(nonzeros(triu(cov(lnEMP(earlywin,:)),1)),[],1));
Z1 = Z / mean(Z);
X1 = X .* Z1;
Y1 = Y .* Z1;
ls_coeff    = inv(X'*X)*X'*Y;
ls_line     = mean(Z1 .* Y) + ls_coeff * X;
figure();
hold on;
scatter(X,Y,Z,'or','MarkerEdgeColor',[8/255,62/255,118/255]);
plot(X,ls_line,'linewidth',1.5,'color',[8/255,62/255,118/255]);
%h = lsline;
%set(h,'color','k')
xlabel('Change in Value Added Covariance, $\Delta Cov(y_{jt},y_{ot})$','interpreter','latex');
ylabel('Change in Employment Covariance $\Delta Cov(l_{jt},l_{ot})$','interpreter','latex');
xlim([-.004 0.002])
ylim([-.004 0.002])
grid on
set(gcf,'color','w')
hold off;

if useHP==0 && dim==37
    print -depsc dcov_scatter.eps
end

if useHP==0 && dim==37
    avgDomar = mean(Domar);
    save datresults_fd.mat lnEMP lnVA lnEMPagg lnVAagg lnInvagg lnCagg VALPcorr earlywin latewin avgDomar VAshavgt
elseif useHP==1 && dim==37
    lnEMPagg_hp = lnEMPagg;
    lnVAagg_hp = lnVAagg;
    lnInvagg_hp = lnInvagg;
    lnCagg_hp = lnCagg;
    VALPcorr_hp = VALPcorr;
    save datresults_hp.mat lnEMPagg_hp lnVAagg_hp lnInvagg_hp lnCagg_hp VALPcorr_hp
end