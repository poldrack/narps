clear all
set more off

global	curr	"../../Scripts/01 - DataProcessing"
global 	raw 	"../../Data/Data Raw"
global 	comb 	"../../Data/Data Raw/_Combined"
global 	proc 	"../../Data/Data Processed"


* **************************************************************************** *
* *** Import Raw Data *** *
* **************************************************************************** *
quietly {

	* Non-Team Members *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	cd "$raw/NonTeamMembers"

	fs *.csv
	foreach f in `r(files)' {
		insheet using "`f'", names
		local   label = substr("`f'", 1, length("`f'") - 4)
		
		* relabel table id's
		cap 	rename	id `label'_id
		
		* identifier for subject pool
		gen		teams = 0
		
		* save temporarily
		save    "../nt_`label'.dta",  replace
		clear 
	}
	cd "../$curr"
	
	
	* Team Members *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	cd "$raw/TeamMembers"

	fs *.csv
	foreach f in `r(files)' {
		insheet using "`f'", names
		local   label = substr("`f'", 1, length("`f'") - 4)
		
		* relabel table id's
		cap 	rename	id `label'_id
		
		* identifier for subject pool
		gen		teams = 1
		
		* save temporarily
		save    "../tm_`label'.dta",  replace
		clear 
	}
	cd "../$curr"
	
	
	* Append Data From Both Pools *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	local tables "asset portfolio position transaction"
	
	foreach f of local tables {
		* append dat sets
		use 	"$raw/nt_`f'.dta"
		append 	using "$raw/tm_`f'.dta"
		
		* save
		save    "$comb/`f'.dta",  replace
		clear 
		
		* drop temporarily created data sets
		erase	"$raw/nt_`f'.dta"
		erase	"$raw/tm_`f'.dta"
	}
	
	
	* Import Team Results Data *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	insheet using "$raw/TeamResults.csv", names
	
	gen		teams = 1
	rename	id teamresults_id
	
	save				"$comb/teamresults.dta", replace 
	export 	delimited	"$comb/teamresults.csv", replace
	clear
}



* **************************************************************************** *
* *** Clean-Up Single Data Tables *** *
* **************************************************************************** *
quietly {

	* Asset Data *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	use 	"$comb/asset.dta"
	drop 	market_id type description
	
	* rename variables
	#delimit ;
	rename	(shares_outcome1 shares_outcome2) 
			(long_outstanding short_outstanding);
	#delimit cr
	
	* hypothesis ID
	egen 	hid = group(asset_id)
	replace	hid = hid - teams * 9
	drop	asset_id
	
	* hypothesis
	gen		hypothesis = substr(name, 6, .)
	replace	hypothesis = subinstr(hypothesis, " | ", ", ", .)
	drop	name
	
	* label variables
	label	var	hid				"Hypothesis ID"
	label	var	hypothesis		"Hypothesis"
	label	var	long_			"Outstanding Shares 'Long'"
	label	var	short_			"Outstanding Shares 'Short'"
	label	var price			"Asset Price"
	label	var	teams			"Subject Pool (0 'NonTeam' / 1 'Team')"
	
	* label values
	label 	define teams 		0 "Non-Team" 1 "Team Members"
	label	values teams teams
	
	* format, order, and sort
	format	%9.3f *_outstanding price
	format	%-82s hypothesis
	order	teams hid hypothesis *_outstanding price
	sort	teams hid
	
	* save/export data
	save				"$comb/asset.dta", replace
	export 	delimited	"$comb/asset.csv", replace
	clear
	
	
	
	* Portfolio Data *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	use 	"$comb/portfolio.dta"
	keep	portfolio_id trader_id balance teams
	
	* rename variables
	rename	(portfolio_id trader_id balance) (aid uid balance_final)
	
	* label variables
	label	var	aid				"Portfolio/Account ID"
	label	var	uid				"Trader/User ID"
	label	var	balance_final	"Final Token Balance"
	label	var	teams			"Subject Pool (0 'NonTeam' / 1 'Team')"
	
	* label values
	label 	define teams 		0 "Non-Team" 1 "Team Members"
	label	values teams teams
	
	* format, order, and sort
	format	%9.3f balance_final
	order	teams aid uid
	sort	teams aid
	
	* save/export data
	save				"$comb/portfolio.dta", replace
	export 	delimited	"$comb/portfolio.csv", replace
	clear

	
			
	* Position Data *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	use 	"$comb/position.dta"
	drop	type
	
	* rename variables
	rename	(position_id portfolio_id) (pid aid)
	
	* hypothesis ID
	egen 	hid = group(asset_id)
	replace	hid = hid - teams * 9
	drop	asset_id
	
	* label variables
	label	var pid				"Position ID"
	label	var	aid				"Portfolio/Account ID"
	label	var	hid				"Hypothesis ID"
	label	var	shares			"Shares Held"
	label	var	teams			"Subject Pool (0 'NonTeam' / 1 'Team')"
	
	* label values
	label 	define teams 		0 "Non-Team" 1 "Team Members"
	label	values teams teams
	
	* format, order, and sort
	format	%9.3f shares
	order	teams pid aid hid
	sort	teams pid
	
	* save/export data
	save				"$comb/position.dta", replace
	export 	delimited	"$comb/position.csv", replace
	clear
	
	

	* Transaction Data *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	use 	"$comb/transaction.dta"
	keep	*_id date cost balance *_price shares teams

	* rename variables
	#delimit ;
	rename	(transaction_id position_id cost old_price new_price) 
			(tid pid investment p_initial p_new);
	#delimit cr
	
	* convert timestamp
	generate double timestamp = clock(date, "YMDhms")
	format 	timestamp %tcCCYY-NN-DD_HH:MM:SS
	drop 	date

	* label variables
	label	var	tid				"Transaction ID"
	label	var	pid				"Position ID"
	label	var	timestamp		"Timestamp"
	label	var	investment		"Invested Tokens (+ 'Long' / - 'Short')"
	label	var	balance			"Token Balance (Before Investment)"
	label	var	p_initial		"Price Before Investment"
	label	var p_new			"Price After Investment"
	label	var shares			"Shares Bought/Sold"
	label	var	teams			"Subject Pool (0 'NonTeam' / 1 'Team')"
	
	* label values
	label 	define teams 		0 "Non-Team" 1 "Team Members"
	label	values teams teams
	
	* format, order, and sort
	format	%9.3f balance investment p_* shares
	order	teams timestamp tid pid balance investment
	sort	teams timestamp

	* save/export data
	save				"$comb/transaction.dta", replace
	export 	delimited	"$comb/transaction.csv", replace
	clear
	
	
	* Results Data *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	use 		"$comb/teamresults.dta"
	
	* reshape data
	#delimit ;
	reshape		long 
				hypo@_decision 
				hypo@_confidence 
				hypo@_similarity, 
				i(teamresults_id) j(hid);
	#delimit cr
	
	* rename variables
	rename		(hypo_* teamresults_id) (* uid)
	
	* replace decision
	replace		decision = "1" if decision == "Yes"
	replace 	decision = "0" if decision == "No"
	destring 	decision, replace
	
	* label variables
	label	var	uid				"Trader/User ID"
	label	var	hid				"Hypothesis ID"
	label	var	decision		"Decision Yes/No"
	label	var confidence		"Self-Rated Confidence Level"
	label	var similarity		"Self-Rated Similarity Level"
	label	var	teams			"Subject Pool (0 'NonTeam' / 1 'Team')"
	
	* label values
	label 	define decision 		0 "no" 1 "yes"
	label	values decision decision
	
	* clean up
	drop if		decision == .
	
	* save/export data
	save				"$proc/TeamResults.dta", replace
	export 	delimited	"$proc/TeamResults.csv", replace
	clear
}



* **************************************************************************** *
* *** Fundamental Values *** *
* **************************************************************************** *
quietly {

	* set number of analysis teams *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	global 	n = 70
	
	* set hypothesis id *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	set 	obs	9
	gen 	hid = _n
	
	* set fundamentals *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	gen		fv = .
	replace fv = 26 / $n	if hid == 1
	replace fv = 15 / $n	if hid == 2
	replace fv = 16 / $n	if hid == 3
	replace fv = 23 / $n	if hid == 4
	replace fv = 59 / $n	if hid == 5
	replace fv = 23 / $n	if hid == 6
	replace fv =  4 / $n	if hid == 7
	replace fv =  4 / $n	if hid == 8
	replace fv =  4 / $n	if hid == 9
	
	* determine confidence intervals *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	gen 	fv_95l = fv - ((fv * (1-fv)) / $n)^0.5 * invnormal(0.975)
	gen 	fv_95u = fv + ((fv * (1-fv)) / $n)^0.5 * invnormal(0.975)
	
	
	* label variables
	label	var	hid			"Hypothesis ID"
	label	var fv			"Fundamental Value (Fraction of Teams)"
	label	var fv_95l		"Lower Bound of 95% CI of Fundamental Value"
	label	var fv_95u		"Upper Bound of 95% CI of Fundamental Value"
	
	* format, order, and sort
	format	%9.3f fv*
	
	* save/export data
	save				"$proc/Fundamentals.dta", replace
	export 	delimited	"$proc/Fundamentals.csv", replace
	clear
	
}

	

* **************************************************************************** *
* *** Merge Data Sets *** *
* **************************************************************************** *
quietly {

	* Asset Holdings *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	#delimit ;
	use 	"$comb/position.dta";
	
	merge 	m:n aid 	using "$comb/portfolio.dta", 
			nogen keepusing(uid);
			
	merge	m:n hid 	using "$comb/asset.dta", 
			nogen keepusing(hypothesis);
	
	drop	aid pid;
	#delimit cr
	
	order	teams uid hid hypothesis shares
	sort	teams uid hid
	
	save				"$proc/Holdings.dta", replace
	export 	delimited	"$proc/Holdings.csv", replace
	clear
	
	
	* Prices *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	use 	"$comb/asset.dta"
	drop	*_outstanding

	save				"$proc/Prices.dta", replace
	export 	delimited	"$proc/Prices.csv", replace
	clear
	
	
	* Transactions *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	#delimit ;
	use 	"$comb/transaction.dta";
	
	merge 	m:n pid 	using "$comb/position.dta", 
			nogen keepusing(aid hid);
			
	merge 	m:n aid 	using "$comb/portfolio.dta", 
			nogen keepusing(uid);
			
	merge	m:n hid 	using "$comb/asset.dta", 
			nogen keepusing(hypothesis);
	
	drop	pid aid;
	drop if	timestamp == .;
	#delimit cr
	
	order	teams hid hypothesis timestamp uid 
	sort	teams hid timestamp 
	
	save				"$proc/Transactions.dta", replace
	export 	delimited	"$proc/Transactions.csv", replace
	clear
	
	
	* Hourly Balanced Panel *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	use 		"$proc/Transactions.dta"
	
	gen 		date = dofc(timestamp)
	gen			day  = day(date)
	gen 		hour = hh(timestamp)
	
	collapse	p_new, by(teams hid day hour)
	fillin 		day hour teams hid
	
	drop 		if day ==  2 & hour < 16
	drop 		if day == 12 & hour > 15
	
	#delimit ;
	merge 		m:n hid 	using "$proc/Fundamentals.dta", 
				nogen keepusing(fv);
	#delimit cr
	
	egen 		time = group(day hour)
	drop		day hour _fillin
	rename		p_new price
	
	order		teams hid time price
	sort		teams hid time

	replace		price = price[_n-1] if price == .
	gen			ae = abs(price - fv)
	
	label 		var time	"Time"
	label 		var price	"Final Market Price"
	label 		var ae		"Absolute Error"
	
	save					"$proc/BalancedPanel.dta", replace
	export 		delimited	"$proc/BalancedPanel.csv", replace
	clear
	
}
