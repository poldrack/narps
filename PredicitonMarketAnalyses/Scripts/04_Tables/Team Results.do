* **************************************************************************** *
* *** Table: Team Results ***                                                  *
* **************************************************************************** *
quietly {
	clear all
	set more off

	global 	proc 				"../../Data/Data Processed"
	use 						"$proc/Holdings.dta"
	merge 	m:n uid hid using 	"$proc/TeamResults.dta", nogen
	merge	m:n hid     using 	"$proc/Fundamentals.dta", nogen
	
	global  file_name 			"TeamResults"
}


* ---------------------------------------------------------------------------- *
* Create Table
* ---------------------------------------------------------------------------- *
quietly {
	
	* keep relevant data
	drop	if teams  == 0
	drop 	if shares == 0
	
	* initialize matrix to store results
	matrix src = J(2,9,.)
	matrix srt = J(5,9,.)
	
	
	
	* Correlation Between Final Holdings and Team Results *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	forvalues i = 1 (1) 9 {
		preserve
		keep if	hid == `i'
			spearman shares decision, stats(rho p)
			matrix src[1,`i'] = r(rho)
			matrix src[2,`i'] = r(p)
		restore
	}
	
	
	
	* Signed-Rank Tests *
	* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
	gen		consistent = 0
	replace consistent = 1	if decision == 1 & shares > 0
	replace consistent = 1 	if decision == 0 & shares < 0
	
	forvalues i = 1 (1) 9 {
		preserve
		keep if	hid == `i'
			signrank	consistent = 0.5
			local 		frac = r(N_pos) / (r(N_pos) + r(N_neg))
			local		p = 2 * (1 - normal(abs(r(z))))
			
			matrix srt[1,`i'] = `frac'
			matrix srt[2,`i'] = r(z)
			matrix srt[3,`i'] = `p'
			
			
			sum	fv		
			local		fv = r(mean) * 100
			sum shares	if consistent == 0
			local		inc = r(mean)
			sum shares	if consistent == 1
			local		con = r(mean)
			
			matrix srt[4,`i'] = `con'
			matrix srt[5,`i'] = `inc'
			
			noi dis		_newline(0)
		restore
	}

}


* ---------------------------------------------------------------------------- *
* Export Table
* ---------------------------------------------------------------------------- *
quietly {

	* to mata...
	mata
		src = st_matrix("src")
		srt = st_matrix("srt")
	end
	
	* matrix to LaTeX
	#delimit ;
		mmat2tex src using "LaTeX/src.tex", 
			rownames(
				"\(\rho_s\)" 
				"\(p\)-value"
			) 
			substitute(
				0.000 "<0.001"
			) 
			fmt(%5.3f) 
			replace;
			
		mmat2tex srt using "LaTeX/srt.tex", 
			rownames(
				"Share of consistent holdings" 
				"\(z\)-value" 
				"\(p\)-value" 
				"Avg. holdings if consistent" 
				"Avg. holdings if inconsistent"
			)
			substitute(
				0.000 "<0.001"
			) 
			insertendrow(
				1 "\midrule" 
				3 "\midrule"
			) 
			fmt(%5.3f) 
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
	cap confirm file   	"$file_name.dvi"
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
	copy  	"$file_name.tex" ///
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
	erase 	"$file_name.tex"
	
	erase 	"LaTeX/src.tex"
	erase	"LaTeX/srt.tex"
	
	clear	all
}

