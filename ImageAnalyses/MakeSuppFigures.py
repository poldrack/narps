#!/usr/bin/env python
# coding: utf-8
"""
Make latex file for supplementary figures

"""

import os
import argparse
import pandas
from narps import Narps

preamble = '''\\documentclass[10pt]{article}
\\usepackage[margin=1in]{geometry}
\\geometry{letterpaper}
\\usepackage{graphicx}
\\usepackage{amssymb}
\\usepackage{epstopdf}
\\usepackage{caption}
\\title{Supplementary Figures}
\\author{Botvinick-Nezer et al.}
\\begin{document}
'''

finale = '\\end{document}\n'


def make_supp_figure_file(narps, figheight=8):
    narps.dirs.get_output_dir('SupplementaryMaterials', base='figures')
    # load supp figure info
    latex = preamble
    info = pandas.read_csv('SuppFiguresInfo.tsv', sep='\t')
    for i in range(info.shape[0]):
        caption = info.loc[i, 'Caption'].replace(
            '#', '\\#').replace('%', '\\%')
        imgfile = os.path.join(
            narps.basedir,
            info.loc[i, 'File']
        )
        latex += '\\begin{figure}[htbp]\n'
        latex += '\\begin{center}\n'
        latex += '\\includegraphics[height=%din]{%s}\n' % (
            info.loc[i, 'Height'], imgfile)
        latex += '\\caption*{Supplementary Figure %d: %s}\n' % (
            info.loc[i, 'Number'],
            caption)
        latex += '\\end{center}\n'
        latex += '\\end{figure}\n'
        latex += '\\newpage\n\n'

    latex += finale

    outfile = os.path.join(
        narps.dirs.dirs['SupplementaryMaterials'],
        'SupplementaryFigures.tex'
    )
    with open(outfile, 'w') as f:
        f.write(latex)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Make latex file for supplementary figures')
    parser.add_argument('-b', '--basedir',
                        help='base directory')
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='use testing mode (no processing)')
    args = parser.parse_args()

    # set up base directory
    if args.basedir is not None:
        basedir = args.basedir
    elif 'NARPS_BASEDIR' in os.environ:
        basedir = os.environ['NARPS_BASEDIR']
        print("using basedir specified in NARPS_BASEDIR")
    else:
        basedir = '/data'
        print("using default basedir:", basedir)

    narps = Narps(basedir)

    if not args.test:
        make_supp_figure_file(narps)
