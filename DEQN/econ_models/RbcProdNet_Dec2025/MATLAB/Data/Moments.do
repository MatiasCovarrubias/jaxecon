clear all
cd "/Users/matiascovarrubias/Library/CloudStorage/Dropbox/Dyn. Prod. Networks/Data"


* Import the CSV
import delimited "merged_data_cleaned.csv", clear
drop if year ==.
* Detrending using Hodrick-Prescott filter
egen sector_id=group(sector)
tsset sector_id year
bys sector: hprescott real_go, stub(hp) smooth(6.25)
bys sector: hprescott real_va, stub(hp) smooth(6.25)
bys sector: hprescott real_ii, stub(hp) smooth(6.25)
bys sector: hprescott employment, stub(hp) smooth(6.25)
bys sector: hprescott real_inv, stub(hp) smooth(6.25)


// gen real_go_hptrend = .
// gen real_go_hpcycle = .
// // replace real_go_hptrend = hp_real_go_sm_1 if sector_id == 1
// // replace real_go_hpcycle = hp_real_go_1 if sector_id == 1
// foreach i of numlist 1/37{
// 	replace real_go_hptrend = hp_real_go_sm_`i' if sector_id == `i'
// 	replace real_go_hpcycle = hp_real_go_`i' if sector_id == `i'
// }

foreach var in real_go real_va real_ii employment real_inv {
	gen `var'_hptrend = .
	gen `var'_hpcycle = .
	foreach i of numlist 1/37{
		replace `var'_hptrend = hp_`var'_sm_`i' if sector_id == `i'
		replace `var'_hpcycle = hp_`var'_`i' if sector_id == `i'
	}
	* Calculate sectoral statistics
	egen vol_`var' = sd(log(`var'_hpcycle)), by(sector)
}
drop hp_*

* Print the summary statistics
summarize vol_real_go vol_real_va vol_real_ii

bys sector: egen corr_va_m = corr(real_va_hpcycle real_ii_hpcycle)
collapse vol* corr_va_m, by(sector)
