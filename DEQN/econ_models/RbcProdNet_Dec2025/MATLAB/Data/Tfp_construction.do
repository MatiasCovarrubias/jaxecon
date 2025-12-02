

************************************************************
**   Housekeeping                                         **
************************************************************

clear all
set more off
set scheme s1color

*cd "C:\Users\bhbltuba\Dropbox\Industry Volatility and Comovement Project\Final Polished Data Code\TFP Construction"
cd "C:\Users\q32292\Documents\DynProdNetworks\DataCleaned\TFP"

************************************************************
**  Import the data                            	  		  **
************************************************************

foreach var in nominal_va real_va employment nominal_inv real_inv_dollars labor_share depreciation_rates nominal_capital real_II real_GO II_shares {

	* Import
	import excel "37 Sector Data.xlsx", sheet(`var') cellrange(A1:AL72) firstrow
	rename A year

	* Rename sectors 
	rename Mining sector_1
	rename Utilities sector_2
	rename Construction sector_3
	rename WoodProducts sector_4
	rename NonmetallicMinerals sector_5
	rename PrimaryMetals sector_6
	rename FabricatedMetals sector_7
	rename Machinery sector_8
	rename ComputerandElectronic sector_9
	rename ElectricalEquipment sector_10
	rename MotorVehicles sector_11
	rename OtherTransequip sector_12
	rename Furnitureandrelated sector_13
	rename MiscMfg sector_14
	rename Foodandbeverage sector_15
	rename Textile sector_16
	rename Apparel sector_17
	rename Paper sector_18
	rename Printing sector_19
	rename Petroleum sector_20
	rename Chemical sector_21
	rename Plastics	 sector_22
	rename WT sector_23
	rename RT sector_24
	rename TW sector_25
	rename Info sector_26
	rename FI sector_27
	rename RE sector_28
	rename ProfBus sector_29
	rename Mgmt sector_30
	rename Admin sector_31
	rename Edu sector_32
	rename Health sector_33
	rename Arts sector_34
	rename Accomm sector_35
	rename FoodServ sector_36
	rename Other sector_37

	* Reshape 
	reshape long sector_, i(year) j(sector)
	
	* Save
	rename sector_ `var'
	save "Output\stata_`var'_37.dta", replace
	
	* Clear 
	clear all
	
}


************************************************************
**  Merge the datasets                        	  		  **
************************************************************
use "Output\stata_nominal_va_37.dta", replace
foreach var in real_va employment nominal_inv real_inv_dollars labor_share depreciation_rates nominal_capital real_II real_GO II_shares {
	merge 1:1 year sector using Output\stata_`var'_37.dta, keep(3) nogen
}


************************************************************
**  Clean up the data	                      	  		  **
************************************************************

* Format year variable
format year %ty

* Set as panel
xtset sector year

replace labor_share = 0.95 if labor_share>0.95

* Compute sector-level average labor shares and depreciation rates (to reduce measurement error in TFP)
bysort sector: egen avg_labor_share = mean(labor_share)
bysort sector: egen avg_depreciation = mean(depreciation_rates)
bysort sector: egen avg_IIshare = mean(II_shares)

gen avg_labor_share_2per = (labor_share[_n]+labor_share[_n - 1])/2
gen avg_II_share_2per = (II_shares[_n]+II_shares[_n - 1])/2

* Compute capital stock using perpetual inventory
gen capital = nominal_capital if year == 1948
bysort sector: replace capital = (1 - depreciation_rates[_n]) * capital[_n - 1] + real_inv_dollars[_n] if year > 1948

* TEMPORARILY SETTING sigma_y : DELETE THIS WHEN LOADING THE REAL CALIBRATED VALUE *
gen sigma_y = 0.8
gen sigma_q = 0.5

* Generate TFP

gen dlog_tfp = ln(real_va[_n]/real_va[_n - 1]) - (1/(1-sigma_y^(-1))) * ln( ((1-avg_labor_share_2per)^(sigma_y^(-1)) * capital[_n]^(1-sigma_y^(-1))+avg_labor_share_2per^(sigma_y^(-1))*employment[_n]^(1-sigma_y^(-1)) ) / ((1-avg_labor_share_2per)^(sigma_y^(-1)) *capital[_n - 1]^(1-sigma_y^(-1))+avg_labor_share_2per^(sigma_y^(-1))*employment[_n - 1]^(1-sigma_y^(-1)) ) )
gen dlog_tfp_smooth = ln(real_va[_n]/real_va[_n - 1]) - (1/(1-sigma_y^(-1))) * ln( ((1-avg_labor_share)^(sigma_y^(-1)) * capital[_n]^(1-sigma_y^(-1))+avg_labor_share^(sigma_y^(-1))*employment[_n]^(1-sigma_y^(-1)) ) / ((1-avg_labor_share)^(sigma_y^(-1)) *capital[_n - 1]^(1-sigma_y^(-1))+avg_labor_share^(sigma_y^(-1))*employment[_n - 1]^(1-sigma_y^(-1)) ) )

gen dlog_tfp_go = ln(real_GO[_n]/real_GO[_n - 1]) - (1/(1-sigma_q^(-1))) * ln( ((1-avg_II_share_2per)^(sigma_q^(-1)) * real_va[_n]^(1-sigma_q^(-1))+avg_II_share_2per^(sigma_q^(-1))*real_II[_n]^(1-sigma_q^(-1)) ) / ((1-avg_II_share_2per)^(sigma_q^(-1)) * real_va[_n - 1]^(1-sigma_q^(-1))+avg_II_share_2per^(sigma_q^(-1))*real_II[_n - 1]^(1-sigma_q^(-1)) ) )

gen dlog_tfp_go_smooth = ln(real_GO[_n]/real_GO[_n - 1]) - (1/(1-sigma_q^(-1))) * ln( ((1-II_shares)^(sigma_q^(-1)) * real_va[_n]^(1-sigma_q^(-1))+II_shares^(sigma_q^(-1))*real_II[_n]^(1-sigma_q^(-1)) ) / ((1-II_shares)^(sigma_q^(-1)) * real_va[_n - 1]^(1-sigma_q^(-1))+II_shares^(sigma_q^(-1))*real_II[_n - 1]^(1-sigma_q^(-1)) ) )

* Reconstruct value added from production function (imposing zero VA TFP shocks)
* VA_rec = CES(K,L) = [(1-alpha)^(1/sigma_y) * K^((sigma_y-1)/sigma_y) + alpha^(1/sigma_y) * L^((sigma_y-1)/sigma_y)]^(sigma_y/(sigma_y-1))
gen va_reconstructed = ((1-avg_labor_share_2per)^(sigma_y^(-1)) * capital^(1-sigma_y^(-1)) + avg_labor_share_2per^(sigma_y^(-1)) * employment^(1-sigma_y^(-1)))^(1/(1-sigma_y^(-1)))
replace va_reconstructed = ((1-labor_share)^(sigma_y^(-1)) * capital^(1-sigma_y^(-1)) + labor_share^(sigma_y^(-1)) * employment^(1-sigma_y^(-1)))^(1/(1-sigma_y^(-1))) if year==1948
gen va_reconstructed_smooth = ((1-avg_labor_share)^(sigma_y^(-1)) * capital^(1-sigma_y^(-1)) + avg_labor_share^(sigma_y^(-1)) * employment^(1-sigma_y^(-1)))^(1/(1-sigma_y^(-1)))

* GO TFP using reconstructed VA (no VA shocks)
gen dlog_tfp_go_noVA = ln(real_GO[_n]/real_GO[_n - 1]) - (1/(1-sigma_q^(-1))) * ln( ((1-avg_II_share_2per)^(sigma_q^(-1)) * va_reconstructed[_n]^(1-sigma_q^(-1))+avg_II_share_2per^(sigma_q^(-1))*real_II[_n]^(1-sigma_q^(-1)) ) / ((1-avg_II_share_2per)^(sigma_q^(-1)) * va_reconstructed[_n - 1]^(1-sigma_q^(-1))+avg_II_share_2per^(sigma_q^(-1))*real_II[_n - 1]^(1-sigma_q^(-1)) ) )

gen dlog_tfp_go_noVA_smooth = ln(real_GO[_n]/real_GO[_n - 1]) - (1/(1-sigma_q^(-1))) * ln( ((1-II_shares)^(sigma_q^(-1)) * va_reconstructed_smooth[_n]^(1-sigma_q^(-1))+II_shares^(sigma_q^(-1))*real_II[_n]^(1-sigma_q^(-1)) ) / ((1-II_shares)^(sigma_q^(-1)) * va_reconstructed_smooth[_n - 1]^(1-sigma_q^(-1))+II_shares^(sigma_q^(-1))*real_II[_n - 1]^(1-sigma_q^(-1)) ) )



replace dlog_tfp = . if year==1948
replace dlog_tfp_smooth = . if year==1948
replace dlog_tfp_go = . if year==1948
replace dlog_tfp_go_smooth = . if year==1948
replace dlog_tfp_go_noVA = . if year==1948
replace dlog_tfp_go_noVA_smooth = . if year==1948

gen TFP = 0
gen TFP_sm = 0
gen TFP_GO = 0
gen TFP_GO_sm = 0
gen TFP_GO_noVA = 0
gen TFP_GO_noVA_sm = 0

replace TFP = 1 if year==1948
replace TFP_sm = 1 if year==1948
replace TFP_GO = 1 if year==1948
replace TFP_GO_sm = 1 if year==1948
replace TFP_GO_noVA = 1 if year==1948
replace TFP_GO_noVA_sm = 1 if year==1948
replace TFP = TFP[_n - 1]*exp(dlog_tfp[_n]) if year>1948
replace TFP_sm = TFP_sm[_n - 1]*exp(dlog_tfp_smooth[_n]) if year>1948
replace TFP_GO = TFP_GO[_n - 1]*exp(dlog_tfp_go[_n]) if year>1948
replace TFP_GO_sm = TFP_GO_sm[_n - 1]*exp(dlog_tfp_go_smooth[_n]) if year>1948
replace TFP_GO_noVA = TFP_GO_noVA[_n - 1]*exp(dlog_tfp_go_noVA[_n]) if year>1948
replace TFP_GO_noVA_sm = TFP_GO_noVA_sm[_n - 1]*exp(dlog_tfp_go_noVA_smooth[_n]) if year>1948


egen mean_dlogTFP_by_sector = mean(dlog_tfp), by(sector)
egen mean_dlogTFP_sm_by_sector = mean(dlog_tfp_smooth), by(sector)
egen mean_dlogTFP_GO_by_sector = mean(dlog_tfp_go), by(sector)
egen mean_dlogTFP_GO_sm_by_sector = mean(dlog_tfp_go_smooth), by(sector)
egen mean_dlogTFP_GO_noVA_by_sector = mean(dlog_tfp_go_noVA), by(sector)
egen mean_dlogTFP_GO_noVA_sm_by_sec = mean(dlog_tfp_go_noVA_smooth), by(sector)

levelsof mean_dlogTFP_by_sector
levelsof mean_dlogTFP_sm_by_sector
levelsof mean_dlogTFP_GO_by_sector
levelsof mean_dlogTFP_GO_sm_by_sector
levelsof mean_dlogTFP_GO_noVA_by_sector
levelsof mean_dlogTFP_GO_noVA_sm_by_sec

* Save
save bea_panel_stata_37.dta, replace


use bea_panel_stata_37, clear
keep year sector TFP
reshape wide TFP*, i(year) j(sector)
save Output\TFP_37.dta, replace
export delimited using Output\TFP_37.csv, replace

use bea_panel_stata_37, clear
keep year sector TFP_sm
reshape wide TFP_sm*, i(year) j(sector)
save Output\TFP_37_sm.dta, replace
export delimited using Output\TFP_37_sm.csv, replace

use bea_panel_stata_37, clear
keep year sector TFP_GO
reshape wide TFP_GO*, i(year) j(sector)
save Output\TFP_GO_37.dta, replace
export delimited using Output\TFP_GO_37.csv, replace

use bea_panel_stata_37, clear
keep year sector TFP_GO_sm
reshape wide TFP_GO_sm*, i(year) j(sector)
save Output\TFP_GO_37_sm.dta, replace
export delimited using Output\TFP_GO_37_sm.csv, replace

use bea_panel_stata_37, clear
keep year sector TFP_GO_noVA
reshape wide TFP_GO_noVA*, i(year) j(sector)
save Output\TFP_GO_noVA_37.dta, replace
export delimited using Output\TFP_GO_noVA_37.csv, replace

use bea_panel_stata_37, clear
keep year sector TFP_GO_noVA_sm
reshape wide TFP_GO_noVA_sm*, i(year) j(sector)
save Output\TFP_GO_noVA_37_sm.dta, replace
export delimited using Output\TFP_GO_noVA_37_sm.csv, replace


