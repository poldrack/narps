texdoc init "$file_name.tex", replace

/***
	\documentclass{standalone}

	\usepackage{booktabs}
	\renewcommand{\arraystretch}{1.2}

	\usepackage{siunitx}
		\sisetup{
			detect-mode,
			tight-spacing			= true,
			group-digits			= false ,
			input-signs				= ,
			input-symbols			= ( ) [ ] - + *,
			input-open-uncertainty	= ,
			input-close-uncertainty	= ,
			table-align-text-post	= false
			}

	\begin{document}
		\input{LaTeX/Content.tex}
	\end{document}
***/
