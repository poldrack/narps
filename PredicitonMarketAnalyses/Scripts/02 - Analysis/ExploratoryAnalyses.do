quietly {
	clear all
	set more off

	global 	raw 	"../../Data/Data Raw"
	global 	proc 	"../../Data/Data Processed"
	
	global   file_name "TeamResults"

	* create log
	log using "../../Results/ExploratoryAnalyses", replace name(text) text
	log using "../../Results/ExploratoryAnalyses", replace name(smcl) smcl
	
	
	* ************************************************************************ *
	* *** Exploratory Analyses *** *
	* ************************************************************************ *
	use 	"$proc/Holdings.dta"
	merge 	m:n uid hid using "$proc/TeamResults.dta", nogen
	merge	m:n hid     using "$proc/Fundamentals.dta", nogen
	
	drop	if teams  == 0
	drop 	if shares == 0
	
	
	
	* Correlation Between Final Holdings and Team Results *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	noi dis		_newline(1)
	noi dis		"Spearman Correlations: Final Holdings vs. Team Result"
	noi dis     "------------------------------------------------------------"
	
	forvalues i = 1 (1) 9 {
		preserve
		keep if	hid == `i'
			spearman shares decision, stats(rho p)
			
			noi dis		"Hypothesis #`i'"
			noi dis		"  rho = " %6.4f r(rho)            ///
						", p = " %6.4f r(p)
			noi dis		_newline(0)
		restore
	}
	
	
	* Signed-Rank Tests *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	noi dis		_newline(1)
	noi dis		"Wilcoxon Tests: Final Holdings Consistent with Team Result"
	noi dis     "------------------------------------------------------------"
	
	gen		consistent = 0
	replace consistent = 1	if decision == 1 & shares > 0
	replace consistent = 1 	if decision == 0 & shares < 0
	
	forvalues i = 1 (1) 9 {
		preserve
		keep if	hid == `i'
			signrank	consistent = 0.5
			local 		frac = r(N_pos) / (r(N_pos) + r(N_neg))
			local		p = 2 * (1 - normal(abs(r(z))))
			
			noi dis		"Hypothesis #`i'"
			noi dis		"  consistent: " %5.3f `frac'      ///
						", z = " %5.3f r(z)                ///
						", p = " %5.3f `p'
			
			
			sum	fv		
			local		fv = r(mean) * 100
			sum shares	if consistent == 0
			local		inc = r(mean)
			sum shares	if consistent == 1
			local		con = r(mean)
			
			noi dis		"  fundamental value:             " %7.2f `fv' "%"
			noi dis		"  avg. holdings if consistent:   " %8.3f `con' 
			noi dis		"  avg. holdings if inconsistent: " %8.3f `inc'
			
			noi dis		_newline(0)
		restore
	}

	
	* close log
	log close _all
}
