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

	* get preferred mode of payment
	clear
	import 	excel using "$pm/Traders - Teams.xlsx", firstrow
	save	"$raw/temp.dta", replace

	clear
	import 	excel using "$pm/Traders - NonTeams.xlsx", firstrow
	append	using "$raw/temp.dta"
	erase	"$raw/temp.dta"

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
	merge	1:1 uid using "$proc/_Payments.dta", nogen
	merge	1:1 username using "$raw/mode.dta", nogen
	erase	"$raw/mode.dta"
	
	
	* depersonalize data
	drop	username ID timestamp name affiliation preferred
	
	* rename variables
	rename	analysisteam teams
	rename	countryofresidence country
	
	* team indicator
	replace teams = "1" if teams == "yes"
	replace teams = "0" if teams == "no"
	destring teams, replace force
	
	label 	define yesno 0 "no" 1 "yes"
	label	values teams yesno
	
	* country/continent of residence
	#delimit ;
	replace	country = strtrim(country);
	replace country = "Belgium" 		
				if 	country == "Belgique";
	replace country = "France"
				if 	country == "FRANCE";
	replace country = "United Kingdom"
				if 	country == "UK" | 
					country == "United Kindom";
	replace country = "United States"
				if 	country == "US" | 
					country == "USA" | 
					country == "Usa" | 
					country == "usa" | 
					country == "United States of America";
	replace country = "Netherlands"
				if 	country == "netherlands";
	
	gen		continent = "";
	replace	continent = "North America"
				if 	country == "United States" | 
					country == "Canada";
	replace continent = "Soth America"
				if 	country == "Brazil";
	replace continent = "Asia"
				if 	country == "Taiwan" | 
					country == "Israel";
	replace continent = "Australia"
				if 	country == "Australia" | 
					country == "New Zealand";
	replace continent = "Europe"
				if 	continent == "";
	#delimit cr
	
	* merge holdings data
	merge	m:n uid using "$proc/Holdings.dta", nogen
	replace shares = abs(shares)
	collapse (mean) pay* y* n* d* t* (sum) shares (first) c* g* pos*, by(uid)
	
	drop if	uid == "1XER09KC"
	gen  active = (shares > 0)
	drop shares
	
	* order and sort
	order	teams active uid gender year* co* position neuro* decision*
	sort	teams active uid 
	
	* label variables
	label	var payment 	"Payment"
	label	var uid			"User/Trader ID"
	label	var active		"Active in Prediction Market"
	label	var	gender		"Gender"
	label	var	yearssince	"Years since PhD"
	label	var yearofbirth	"Year of Birth"
	label	var	country		"Country of Residence"
	label	var continent	"Continent of Residence"
	label	var	position	"Academic Position"
	label	var neuro		"Self-reported Expertise in Neuroimaging"
	label	var decision	"Self-reported Expertise in Decision Sciences"

	* save
	save				"$proc/_Demographics.dta", replace
	export 	delimited	"$proc/_Demographics.csv", replace
}
