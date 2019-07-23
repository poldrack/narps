texdoc init "$file_name.tex", replace

/***
\documentclass{standalone}

\usepackage{array}
\usepackage{mathtools}
\usepackage{booktabs}
\usepackage{threeparttable}
\renewcommand{\arraystretch}{1.2}

\usepackage{dcolumn}
\newcolumntype{d}[1]{D{.}{.}{#1}}
\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}


\begin{document}
\begin{threeparttable}
	\begin{tabular}{l *{9}{d{1.3}}}
	
		\multicolumn{9}{l}{\textit{\textbf{Panel A.} 
		                   Rank correlations (final holdings---team results)}}\\[0.25em]
		\toprule
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		\textit{Hypothesis}                                                   &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#1}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#2}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#3}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#4}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#5}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#6}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#7}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#8}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#9}}                               \\
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		\midrule
		\input{LaTeX/src.tex}
		\bottomrule															  \\[0.50em]
		
		
		\multicolumn{9}{l}{\textit{\textbf{Panel B.}
						   Final holdings consistent with team results}}      \\[0.25em]
		\toprule
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		\textit{Hypothesis}                                                   &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#1}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#2}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#3}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#4}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#5}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#6}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#7}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#8}}                               &
		\multicolumn{1}{M{1.5cm}}{\textbf{\#9}}                               \\
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		\midrule
		\input{LaTeX/srt.tex}
		\bottomrule
	\end{tabular}
		
\end{threeparttable}
\end{document}
***/
