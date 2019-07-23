texdoc init "$file_name.tex", replace

/***
\documentclass{standalone}

\usepackage{array}
\usepackage{booktabs}
\usepackage{threeparttable}
\renewcommand{\arraystretch}{1.2}

\usepackage{dcolumn}
\newcolumntype{d}[1]{D{.}{.}{#1}}
\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}


\begin{document}
\begin{threeparttable}
	\begin{tabular}{c l c 
					d{2.2} d{2.2}  
					d{3.0} d{3.0} c
					d{2.2} d{2.2}
					d{3.0} d{3.0}}
	
		\toprule
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		&&&
		\multicolumn{4}{c}{\textbf{``Non-Teams''}}                            &&
		\multicolumn{4}{c}{\textbf{``Teams''}}                                \\
		\cmidrule{4-7}
		\cmidrule{9-12}
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		\textbf{ID}                                                           &
		\textbf{Hypothesis$^\ast$}                                            &&
		\multicolumn{1}{M{1.5cm}}{\textit{Tokens Invested}$^\dagger$}         &
		\multicolumn{1}{M{1.5cm}}{\textit{Volume (Shares)}$^\ddagger$}        &
		\multicolumn{1}{M{1.2cm}}{\textit{Trans\-actions}}                    &
		\multicolumn{1}{M{1.2cm}}{\textit{No. of Traders}}                    &&
		\multicolumn{1}{M{1.5cm}}{\textit{Tokens Invested}$^\dagger$}         &
		\multicolumn{1}{M{1.5cm}}{\textit{Volume (Shares)}$^\ddagger$}        &
		\multicolumn{1}{M{1.2cm}}{\textit{Trans\-actions}}                    &
		\multicolumn{1}{M{1.2cm}}{\textit{No. of Traders}}                    \\
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		\midrule
		\input{LaTeX/Content.tex}
		\bottomrule
	\end{tabular}
	
	\begin{tablenotes}[flushleft]
		\small
		\item $^\ast$ 	  $ER$ and $EI$ denote the \textit{``Equal Range''} and
						  \textit{``Equal Indifference''} condition, respectively.
		\item $^\dagger$  Mean number of tokens (points) invested per transaction.
		\item $^\ddagger$ Mean number of shares bought or sold per transaction.
	\end{tablenotes}
		
\end{threeparttable}
\end{document}
***/
