quietly {
	clear all
	set more off

	global 	raw 	"../../Data/Data Raw"
	global 	proc 	"../../Data/Data Processed"

	* create log
	log using "../../Results/MainAnalyses", replace name(text) text
	log using "../../Results/MainAnalyses", replace name(smcl) smcl
	
	
	* ************************************************************************ *
	* *** Main Analyses *** *
	* ************************************************************************ *
	preserve
	use 	"$proc/Prices.dta"
	merge	m:1 hid using "$proc/Fundamentals.dta", nogen
	reshape	wide price@, i(hid) j(teams)
	

	* Correlation Between Prices *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	noi dis		_newline(1)
	noi dis		"Spearman Correlations:"
	noi dis     "------------------------------------------------------------"
	
	spearman	price0 fv, stats(rho p)
	noi dis		"Fundamental Value vs. Market Belief 'Non-Teams'"
	noi dis		"  rho = " %5.3f r(rho)                    ///
				", p = " %5.3f r(p)
	noi dis		_newline(0)
	
	spearman	price1 fv, stats(rho p)
	noi dis		"Fundamental Value vs. Market Belief 'Teams'"
	noi dis		"  rho = " %5.3f r(rho)                    ///
				", p = " %5.3f r(p)
	noi dis		_newline(0)
	
	spearman	price*, stats(rho p)
	noi dis		"Market Belief 'Non-Teams' vs. Market Belief 'Teams'"
	noi dis		"  rho = " %5.3f r(rho)                    ///
				", p = " %5.3f r(p)
	noi dis		_newline(0)
	
	
	* Signed-Rank Tests *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	noi dis		_newline(1)
	noi dis		"Wilcoxon Signed-Rank Tests:"
	noi dis     "------------------------------------------------------------"
	
	signrank 	price0 = fv
	noi dis		"Fundamental Value vs. Market Belief 'Non-Teams'"
	local		p = 2 * (1 - normal(abs(r(z))))
	noi dis		"  z = " %5.3f r(z)                        ///
				", p = " %5.3f `p'
	noi dis		_newline(0)
	
	signrank 	price1 = fv
	noi dis		"Fundamental Value vs. Market Belief 'Teams'"
	local		p = 2 * (1 - normal(abs(r(z))))
	noi dis		"  z = " %5.3f r(z)                        ///
				", p = " %5.3f `p'
	noi dis		_newline(0)
	
	signrank 	price0 = price1
	noi dis		"Market Belief 'Non-Teams' vs. Market Belief 'Teams'"
	local		p = 2 * (1 - normal(abs(r(z))))
	noi dis		"  z = " %5.3f r(z)                        ///
				", p = " %5.3f `p'
	noi dis		_newline(0)
	
	restore
	
	* close log
	log close _all
}
