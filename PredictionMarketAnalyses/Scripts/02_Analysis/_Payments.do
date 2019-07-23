clear all
set more off

global 	raw 	"../../Data/Data Raw"
global 	proc 	"../../Data/Data Processed"
global	pm		"../../../Prediction Markets"
global	dl		"../../Data/_m4rk3ts Downloads"


* **************************************************************************** *
* *** Determine Payments *** *
* **************************************************************************** *
quietly {
	use 	"$proc/Holdings.dta"
	merge	m:n hid using "$proc/Fundamentals.dta", nogen

	gen		payment = .
	format	payment %9.2f 
	replace	payment = 0.5 * (shares * fv)				if shares > 0
	replace payment = 0.5 * (abs(shares) * (1 - fv))	if shares < 0

	* collapse data
	collapse (sum) payment, by(uid)
	
	* cap payments at $200
	replace payment = 200                               if payment > 200
	
	* save/export data
	save	"$raw/Payments.dta", replace
	clear
}

* **************************************************************************** *
* *** Prepare Payment List *** *
* **************************************************************************** *
quietly {

	* get preferred mode of payment
	clear
	import 	excel using "$pm/Traders - Teams.xlsx", firstrow
	save	"$raw/temp.dta", replace

	clear
	import 	excel using "$pm/Traders - NonTeams.xlsx", firstrow
	append	using "$raw/temp.dta"
	erase	"$raw/temp.dta"

	keep	preferredmodeofpayment email
	rename	preferredmodeofpayment mode
	rename	email username
	save	"$raw/mode.dta", replace


	* get trader e-mails
	clear
	insheet using "$dl/NonTeamMembers/trader.csv", names
	save	"$raw/temp.dta", replace

	clear
	insheet using "$dl/TeamMembers/trader.csv", names

	append 	using "$raw/temp.dta"
	erase	"$raw/temp.dta"

	drop	password
	rename	id uid
	duplicates drop

	* merge payments with e-mails and preferred mode of payment
	merge	1:1 uid using "$raw/Payments.dta", nogen
	merge	1:1 username using "$raw/mode.dta", nogen
	erase	"$raw/mode.dta"
	erase	"$raw/payments.dta"

	drop 	if payment == 0
	
	* label variables
	label 	var uid			"User/Trader ID"
	label	var username	"User Name"
	label	var payment		"Payment"
	label	var mode		"Preferred Mode of Payment"
	
	* save
	save				"$proc/_Payments.dta", replace
	export 	delimited	"$proc/_Payments.csv", replace
}
