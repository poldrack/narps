* **************************************************************************** *
* *** Table: Prediction Market Details ***                                     *
* **************************************************************************** *
quietly {
	clear all
	set more off

	use 	"../../Data/Data Processed/Transactions.dta"
	global  file_name "MarketDetails"
}


* ---------------------------------------------------------------------------- *
* Create Table
* ---------------------------------------------------------------------------- *
quietly {

	* hypotheses
	replace hypothesis = "Pos. Effect, Gains, VMPFC, EI" 					 ///
		if hid == 1
	replace hypothesis = "Pos. Effect, Gains, VMPFC, ER" 					 ///
		if hid == 2
	replace hypothesis = "Pos. Effect, Gains, Ventral Striatum, EI" 		 ///
		if hid == 3
	replace hypothesis = "Pos. Effect, Gains, Ventral Striatum, ER" 		 ///
		if hid == 4
	replace hypothesis = "Neg. Effect, Losses, VMPFC, EI" 					 ///
		if hid == 5
	replace hypothesis = "Neg. Effect, Losses, VMPFC, ER" 					 ///
		if hid == 6
	replace hypothesis = "Pos. Effect, Losses, Amygdala, EI" 				 ///
		if hid == 7
	replace hypothesis = "Pos. Effect, Losses, Amygdala, ER" 				 ///
		if hid == 8
	replace hypothesis = "Greater Pos. Effect, Losses, Amygdala, ER vs. EI"  ///
		if hid == 9
	
	* absolute number of shares/tokens
	replace shares  	= abs(shares)
	replace investment 	= abs(investment)
	
	* determine number of traders
	gen   traders = .
	forvalues j = 0 (1) 1 {
		forvalues k = 1 (1) 9 {
			unique  uid           		if teams == `j' & hid == `k'
			replace traders = r(unique) if teams == `j' & hid == `k'
		}
	}
	
	* collapse and reshape
	#delimit ;
		collapse 
			(mean)  tokens  = investment 
			(mean)  volume  = shares
			(count) trades  = tid
			(mean)  traders = traders,
			by(hid hypothesis teams);
	
		reshape wide 
			tokens volume trades traders, 
			i(hid) j(teams);
	#delimit cr

}


* ---------------------------------------------------------------------------- *
* Export Table
* ---------------------------------------------------------------------------- *
quietly {
	
	#delimit ;
		* varlist
		gen     space = .
		local   varlist "
				hid hypothesis space
				tokens0 volume0	trades0 traders0 space
				tokens1 volume1	trades1 traders1
				";
	
		* export table contents as .tex
		cap ssc  	install listtex;
		listtex  	`varlist' using "LaTeX/Content.tex", 
					begin("") delimiter("&") end(`"\\"') missnum("")
					replace;

		* create .tex table
		cap ssc  	install texdoc;
		texdoc   	do "LaTeX/$file_name - LaTeX.do";
	#delimit cr

	* call LaTeX, dvi2ps, and ps2pdf
	shell latex 		"$file_name.tex"
	shell dvips -P pdf 	"$file_name.dvi"
	shell ps2pdf 		"$file_name.ps"
	
	* confirm LaTeX has compiled properly
	cap confirm file    "$file_name.dvi"
	if _rc != 0 {
		noi di _n
		noi di _col(40) "ERROR"
		noi di _col(15) _dup(55) "~"
		noi di _col(15) "The table has not been properly compiled using LaTeX."
		noi di _col(15) "Either there is no TeX distribution installed on your"
		noi di _col(15) "computer or the called .exe files are not part of the"
		noi di _col(15) "PATH environmental variable. Thus, the table was not" 
		noi di _col(15) "exported in .ps and .pdf format."
		noi di _col(15) _dup(55) "~"
		exit
	}

	* move compiled files to <Tables> folder
	copy  	"LaTeX/Content.tex" ///
			"../../Tables/$file_name.tex",  replace
	copy  	"$file_name.ps" ///
			"../../Tables/$file_name.ps",  replace
	copy  	"$file_name.pdf" ///
			"../../Tables/$file_name.pdf", replace
	
	* drop temporary files
	erase 	"$file_name.aux"
	erase 	"$file_name.dvi"
	erase 	"$file_name.log"
	erase 	"$file_name.pdf"
	erase 	"$file_name.ps"
	erase	"$file_name.tex"
	erase 	"LaTeX/Content.tex"
	
	clear	all
}
