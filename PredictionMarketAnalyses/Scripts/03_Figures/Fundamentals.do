* **************************************************************************** *
* *** Figure: Prediction Market Beliefs ***                                    *
* **************************************************************************** *
quietly {
	clear  all
	set    more   off
	set    scheme s1mono

	* import data
	insheet using 		"../../Data/Data Raw/Fundamentals.csv"
	merge 1:1 hid using	"../../Data/Data Processed/Fundamentals.dta", nogen

	* get colors
	do     "_Colors.do"
	
	* set font scheme
	graph set window fontface default

	* gen hypothesis id for sorting
	sort	fv
	gen		id =  _n

	forvalues i = 1 (1) 9 {
		local hyp = hid[`i']
		label define ids `i' "#`hyp'", add
	}
	label values id ids


	* ------------------------------------------------------------------------ *
	* Panel A: Final Market Prices
	* ------------------------------------------------------------------------ *
	#delimit ;
	twoway (
		rcap fv_95l fv_95u id,
			lwidth(vthin)
		)(
		scatter fv id,
			msymbol(d)
			mfcolor(gs11)
			mlcolor(gs7)
			mlwidth(thin)
			msize(2.25)
		)(
		scatter fv_active id,
			msymbol(o)
			mfcolor("$fc_m1")
			mlcolor("$lc_m1")
			mlwidth(thin)
			msize(3.25)
			// ::::::::::::::::::::::::::::::::::: //
			ytitle("Fraction of Teams",
				size(3.0)
				margin(r=1)
				just(left)
			)
			yscale(range(-0.05 1.05))
			ylabel(0.00 (0.10) 1.00,
				labsize(2.75)
				labgap(vsmall)
				angle(45)
				format(%9.2f)
				grid gmin gmax
				glwidth(vvthin)
				glcolor(gs12)
				glpattern(solid)
			)
			// ::::::::::::::::::::::::::::::::::: //
			xtitle("Hypothesis ID",
				size(3.0)
				margin(t=2)
			)
			xscale(range(0.5 9.5))
			xlabel(1 (1) 9,
				valuelabels 
				labsize(2.75)
				labgap(vsmall)
				angle(45)
				grid
				glwidth(vvthin)
				glcolor(gs12)
				glpattern(solid)
			)
			// ::::::::::::::::::::::::::::::::::: //
			plotregion(
				lcolor(black)
				lwidth(thin)
				margin(small)
			)
			graphregion(
				margin(medium)
			)
			// ::::::::::::::::::::::::::::::::::: //
			legend(
				order(
					2 "All Teams"
					1 "95% CI (All Teams)"
					3 "Active Traders"
				)
				ring(0)
				bplacement(nw)
				rows(4) cols(1)
				size(2.5)
				symxsize(2.75)
				symysize(1.2)
				rowgap(1.2)
				forcesize
				bmargin(small)
			)
			xsize(20)
			ysize(15)
		);
	#delimit cr


	* ------------------------------------------------------------------------ *
	* Export Figure
	* ------------------------------------------------------------------------ *
	local file_path "../../Figures/Fundamentals"
	graph save      "`file_path'.gph", replace
	graph export    "`file_path'.eps", replace
	graph export    "`file_path'.ps",  replace
	graph export    "`file_path'.pdf", replace
	graph export    "`file_path'.png", replace width(3000)

	window manage close graph
	clear all
}
