* **************************************************************************** *
* *** Figure: Prediction Market Beliefs ***                                    *
* **************************************************************************** *
quietly {
	clear  all
	set    more   off
	set    scheme s1mono

	use    					"../../Data/Data Processed/Prices.dta"
	merge  m:1 hid using 	"../../Data/Data Processed/Fundamentals.dta", nogen

	* get colors
	do     "_Colors.do"
	
	* set font scheme
	graph set window fontface default


	* reshape data
	reshape wide price, i(hid) j(teams)

	* gen hypothesis id for sorting
	sort	fv price1
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
			msize(1.75)
		)(
		scatter price0 id,
			msymbol(o)
			mfcolor("$fc_c1")
			mlcolor("$lc_c1")
			mlwidth(thin)
			msize(2.75)
		)(
		scatter price1 id,
			msymbol(o)
			mfcolor("$fc_m1")
			mlcolor("$lc_m1")
			mlwidth(thin)
			msize(2.75)
			// ::::::::::::::::::::::::::::::::::: //
			title("{bf:Panel A}",
				size(3.0)
				margin(b=2)
			)
			// ::::::::::::::::::::::::::::::::::: //
			ytitle("Final Market Prices",
				size(2.5)
				margin(r=1)
				just(left)
			)
			yscale(range(-0.05 1.05))
			ylabel(0.00 (0.10) 1.00,
				labsize(2.25)
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
				size(2.5)
				margin(t=3.8)
			)
			xscale(range(0.5 9.5))
			xlabel(1 (1) 9,
				valuelabels 
				labsize(2.25)
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
					2 "Fundamental Value"
					1 "95% Confidence Interval"
					3 "Non-Team Members"
					4 "Team Members"
				)
				ring(0)
				bplacement(nw)
				rows(4) cols(1)
				size(2.0)
				symxsize(2.5)
				symysize(1.2)
				rowgap(0.9)
				forcesize
				bmargin(small)
			)
			xsize(18)
			ysize(20)
			nodraw
			name(a, replace)
		);
	#delimit cr



	* ------------------------------------------------------------------------ *
	* Panel B: Final Market Prices vs. Fundamental Values
	* ------------------------------------------------------------------------ *
	set 	obs 101
	gen		e = (_n-1) / 100
	gen		ci_lb = e - ((e * (1-e)) / 70)^0.5 * invnormal(0.975)
	gen		ci_ub = e + ((e * (1-e)) / 70)^0.5 * invnormal(0.975)


	#delimit ;
	twoway (
		rarea ci_lb ci_ub e,
			lwidth(vthin)
			lcolor(gs12)
			fcolor(gs15)
			lalign(center)
		)(
		function y=x,
			range(0 1)
			lpattern(solid)
			lwidth(thin)
			lcolor(gs9)
		)(
		scatter price0 fv,
			msymbol(o)
			msize(2.75)
			mfcolor("$fc_c2")
			mlcolor("$lc_c2")
			mlwidth(thin)
		)(
		lfit price0 fv,
			lcolor("$lc_c2")
			lpattern(shortdash)
			lwidth(medthin)
			range(0 1)
		)(
		scatter price1 fv,
			msymbol(o)
			msize(2.75)
			mfcolor("$fc_m2")
			mlcolor("$lc_m2")
			mlwidth(thin)
		)(
		lfit price1 fv,
			lcolor("$lc_m2")
			lpattern(shortdash)
			lwidth(medthin)
			range(0 0.66)
			// ::::::::::::::::::::::::::::::::::: //
			title("{bf:Panel B}",
				size(3.0)
				margin(b=2)
			)
			// ::::::::::::::::::::::::::::::::::: //
			xtitle("Fundamental Values",
				margin(0 0 0 2)
				size(2.5)
			)
			xscale(range(-0.05 1.05))
			xlabel(0.00 (0.10) 1.00,
				format(%9.2f)
				angle(45)
				labsize(2.25)
				labgap(vsmall)
				grid
				glpattern(solid)
				glwidth(vvthin)
				glcolor(gs12)
			)
			// ::::::::::::::::::::::::::::::::::: //
			ytitle("Final Market Prices",
				margin(r=1)
				size(2.5)
			)
			yscale(range(-0.05 1.05))
			ylabel(0.00 (0.10) 1.00,
				format(%9.2f)
				angle(45)
				labsize(2.25)
				labgap(vsmall)
				grid
				glpattern(solid)
				glwidth(vvthin)
				glcolor(gs12)
				gmin gmax
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
					3 "Non-Team Members"
					4 "Linear Fit 'Non-Teams'"
					5 "Team Members"
					6 "Linear Fit 'Teams'"
					2 "45° Line"
					1 "95% Confidence Interval"
				)
				ring(0)
				bplacement(se)
				rows(6) cols(1)
				size(2.0)
				symxsize(2.5)
				symysize(1.2)
				rowgap(0.9)
				forcesize
				bmargin(small)
			)
			xsize(18)
			ysize(20)
			nodraw
			name(b, replace)
		);
	#delimit cr


	* ------------------------------------------------------------------------ *
	* Combine Panels
	* ------------------------------------------------------------------------ *
	#delimit ;
	graph combine a b,
		iscale(*1.25)
		xsize(20)
		ysize(12);
	#delimit cr

	
	* ------------------------------------------------------------------------ *
	* Export Figure
	* ------------------------------------------------------------------------ *
	local file_path "../../Figures/MarketBeliefs"
	graph save      "`file_path'.gph", replace
	graph export    "`file_path'.eps", replace
	graph export    "`file_path'.ps",  replace
	graph export    "`file_path'.pdf", replace
	graph export    "`file_path'.png", replace width(3000)

	window manage close graph
	clear all
}
