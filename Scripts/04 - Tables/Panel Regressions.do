* **************************************************************************** *
* *** Table: Panel Regressions ***                                             *
* **************************************************************************** *
quietly {
	clear all
	set more off

	use 	"../../Data/Data Processed/BalancedPanel.dta"
	global  file_name "PanelRegressions"
}


* ---------------------------------------------------------------------------- *
* Create Table
* ---------------------------------------------------------------------------- *
quietly {

	* model (1)
	reg			ae c.time i.teams, vce(robust)
	estimates 	store m1
	
	* model (2)
	reg			ae c.time##i.teams, vce(robust)
	estimates 	store m2
	
	* write to LaTeX
	#delimit ;
	noi esttab m1 m2 using "LaTeX/Content.tex", 
		nobaselevels 
		collabels(none)
		cells(
			b(star fmt(3)) 
			se(par fmt(3))
		)
		star(
			* 0.05  ** 0.005
		)
		stats(
			r2_a N, 
				fmt(3 0) 
				labels(
					`"\$Adj.\ R^2\$"'
					`"Observations"'
				)
				layout(
					"\multicolumn{1}{S[table-format=1.3]}{@}"
					"\multicolumn{1}{c}{@}"
				)
		)
		label varlabels(_cons Constant) 
		varwidth(40) modelwidth(15)
		booktabs alignment(S[table-format=1.3])
		interaction(" $\times$ ")
		replace;
	#delimit cr
	

	* create .tex table
	cap ssc  	install texdoc;
	texdoc   	do "LaTeX/$file_name - LaTeX.do";
	
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
	erase 	"$file_name.tex"
	erase 	"LaTeX/Content.tex"
	
	clear 	all
}
