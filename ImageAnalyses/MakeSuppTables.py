#!/usr/bin/env python
# coding: utf-8
"""
Make latex file for supplementary figures

"""

import os
import argparse
import pandas
import numpy
from narps import Narps

preamble = '''\\documentclass[10pt]{article}
\\usepackage[margin=0.5in]{geometry}
\\geometry{letterpaper}
\\usepackage{graphicx}
\\usepackage{amssymb}
\\usepackage{epstopdf}
\\usepackage{booktabs}
\\usepackage{caption}
\\title{Supplementary Tables}
\\author{Botvinick-Nezer et al.}
\\begin{document}
'''

finale = '\\end{document}\n'


def make_supp_table_file(narps, default_cwidth=2):
    narps.dirs.get_output_dir('SupplementaryMaterials', base='figures')
    # load supp figure info
    latex = preamble
    info = pandas.read_csv(
        'SuppTablesInfo.tsv',
        sep='\t', index_col=False)
    for i in range(info.shape[0]):
        # check if it's an image file
        if not isinstance(info.loc[i, 'File'], str) or\
                not isinstance(info.loc[i, 'Caption'], str):
            print('skipping table:')
            print(info.loc[i, :])
            continue

        caption = info.loc[i, 'Caption'].replace(
            '#', '\\#').replace('%', '\\%')
        tblfile = info.loc[i, 'File']

        if tblfile.find('png') > -1:
            tblfile = os.path.join(
                narps.basedir,
                tblfile
            )
            latex += '\\begin{figure}[htbp]\n'
            latex += '\\begin{center}\n'
            latex += '\\includegraphics[height=%din]{%s}\n' % (
                8, tblfile)
            latex += '\\caption*{Supplementary Table %d: %s}\n' % (
                info.loc[i, 'Table'],
                caption)
            latex += '\\end{center}\n'
            latex += '\\end{figure}\n\n\n'
        else:
            # otherwise it's a table
            tblfile = os.path.join(
                narps.basedir,
                tblfile
            )
            if tblfile.find('.tsv') > -1:
                sep = '\t'
            else:
                sep = ','
            tbl = pandas.read_csv(tblfile, sep=sep)
            # set up width formatting

            if isinstance(info.loc[i, 'Width'], str):
                cwidth = [float(i) for i in info.loc[i, 'Width'].split(',')]
                # use default if size is mismatched
                if len(cwidth) != tbl.shape[1]:
                    print(i, 'width mismatch, using default')
                    cwidth = [default_cwidth for i in range(tbl.shape[1])]

            elif isinstance(info.loc[i, 'Width'], float):
                if numpy.isnan(info.loc[i, 'Width']):
                    cwidth = [default_cwidth for i in range(tbl.shape[1])]
                else:
                    cwidth = [info.loc[
                            i, 'Width'] for i in range(tbl.shape[1])]
            cformat = '|'.join([
                'p{%f cm}' % i for i in cwidth])
            print(cformat)
            latex += '\\begin{table}\n'
            latex += '\\caption*{Supplementary Table %d: %s}\n' % (
                info.loc[i, 'Table'],
                caption)
            with pandas.option_context("max_colwidth", 1000):
                t = tbl.to_latex(index=False,
                                 float_format="{:0.3f}".format,
                                 na_rep='',
                                 column_format=cformat).replace(
                                        'nan', '').replace(
                                            'R\\textasciicircum 2',
                                            '$R^2$').replace(
                                                '\$', '$') # noqa, flake8 issue

            print(t)
            latex += t
            latex += '\\end{table}\n\n\n'
        latex += '\\clearpage\n\n'

    latex += finale

    outfile = os.path.join(
        narps.dirs.dirs['SupplementaryMaterials'],
        'SupplementaryTables.tex'
    )
    with open(outfile, 'w') as f:
        f.write(latex)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Make latex file for supplementary tables')
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
        make_supp_table_file(narps)
