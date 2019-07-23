* **************************************************************************** *
* *** Figure: Time Series of Market Prices ***                                 *
* **************************************************************************** *
quietly {
	clear  all
	set    more   off
	set    scheme s1mono

	* import data
	use    "../../Data/Data Processed/BalancedPanel.dta"

	* get colors
	do     "_Colors.do"
	
	* set font scheme
	graph set window fontface default


	* reshape data
	reshape wide price ae, i(hid time) j(teams)
	
	forvalues i = 1 (1) 9 {
		preserve
		keep if hid == `i'
		
		* y-axis title and labels *
		* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
		if `i' == 1 | `i' == 4 | `i' == 7 {
			#delimit ;
			local yaxis "
				ytitle("Avg. Price per Hour",
					size(2.75)
					margin(0 2 0 0)
				)
				ylabel(0.00 (0.25) 1.00,
					labsize(2.5)
					labgap(1)
					angle(45)
					format(%9.2f)
					grid gmin gmax
					glwidth(vvthin)
					glcolor(gs11)
					glpattern(solid)
				)
				fxsize(85)
			";
			#delimit cr
		}
		else {
			#delimit ;
			local yaxis "
				ytitle("",
					size(2.75)
					margin(0 2 0 0)
				)
				ylabel(0.00 (0.25) 1.00,
					nolabel
					grid gmin gmax
					glwidth(vvthin)
					glcolor(gs11)
					glpattern(solid)
				)
				fxsize(70)
			";
			#delimit cr
		}
		
		* x-axis title and labels *
		* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
		if `i' == 7 | `i' == 8 | `i' == 9 {
			#delimit ;
			local xaxis "
				xtitle("Time (in Hours)",
					size(2.75)
					margin(0 0 0 1)
				)
				xlabel(0 (48) 240,
					labsize(2.5)
					labgap(1)
					angle(45)
					grid
					glwidth(vvthin)
					glcolor(gs10)
					glpattern(solid)
				)
				fysize(70)
			";
			#delimit cr
		}
		else {
			#delimit ;
			local xaxis "
				xtitle("",
					size(2.75)
					margin(0 0 0 1)
				)
				xlabel(0 (48) 240,
					nolabel
					grid
					glwidth(vvthin)
					glcolor(gs10)
					glpattern(solid)
				)
				fysize(60)
			";
			#delimit cr
		}
	
	
		* individual graphs * 
		* :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: *
		#delimit ;
		twoway (
			line price0 price1 fv time, 
				subtitle("{bf:{it:Hypothesis #`i'}}", 
					size(2.75)
					margin(b=2)
				)
				lcolor(
					"$lc_c1" "$lc_m1" gs4
				)
				lpattern(
					solid solid shortdash
				)
				lwidth(
					0.25 0.25 0.25
				)
			),
			// ::::::::::::::::::::::::::::::::::: //
			`yaxis'
			ymlabel(0.00 (0.05) 1.00,
				grid
				labcolor(white)
				glwidth(vvvthin)
				glcolor(gs12)
				glpattern(solid)
			)
			// ::::::::::::::::::::::::::::::::::: //
			`xaxis'
			xmlabel(0 (12) 240,
				grid 
				labcolor(white)
				glwidth(vvvthin)
				glcolor(gs12)
				glpattern(solid)
			)
			// ::::::::::::::::::::::::::::::::::: //
			plotregion(
				margin(small)
				lcolor(black)
				lwidth(thin)
			)
			graphregion(
				margin(vsmall)
			)
			// ::::::::::::::::::::::::::::::::::: //
			legend(
				order(
					3 " Fundamental Value"
					1 " 'Non-Teams'"
					2 " 'Teams'"
				)
				cols(3)
				size(2.05)
				symxsize(large)
				symysize(small)
				bmargin(t=3 r=6.5)
				height(1.5)
				lwidth(vvvthin)
				lcolor(gs14)
			)
			// ::::::::::::::::::::::::::::::::::: //
			nodraw
			name(g`i', replace);
		#delimit cr
		restore
	}
	
	
	* ------------------------------------------------------------------------ *
	* Combine Panels
	* ------------------------------------------------------------------------ *
	#delimit ;
	grc1leg g1 g2 g3 g4 g5 g6 g7 g8 g9,
		cols(3)
		iscale(0.75)
		position(5)
		legendfrom(g1)
		xsize(10)
		ysize(20)
		name(combined, replace)
	;
	
	graph display combined,	
		xsize(18) ysize(20);
	#delimit cr


	* ------------------------------------------------------------------------ *
	* Export Figure
	* ------------------------------------------------------------------------ *
	local file_path "../../Figures/TimeSeries"
	graph save      "`file_path'.gph", replace
	graph export    "`file_path'.eps", replace
	graph export    "`file_path'.ps",  replace
	graph export    "`file_path'.pdf", replace
	graph export    "`file_path'.png", replace width(3000)

	window manage close graph
	clear all
}
