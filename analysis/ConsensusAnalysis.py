#!/usr/bin/env python
# coding: utf-8
"""
run consensus analysis to identify overall pattern
analysis method developed by T Nichols and J Mumford
"""


import os
import sys
import glob
import numpy
import nibabel
import nilearn.plotting
import nilearn.input_data
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import scipy.stats
from narps import Narps, hypnums, hypotheses
from narps import NarpsDirs # noqa, flake8 issue
from utils import log_to_file


def t_corr(y, res_mean=None, res_var=None, Q=None):
    """
    perform a one-sample t-test on correlated data
    y = data (n observations X n vars)
    res_mean = Common mean over voxels and results
    res_var  = Common variance over voxels and results
    Q = "known" correlation across observations
    - (use empirical correlation based on maps)
    """

    npts = y.shape[0]
    X = numpy.ones((npts, 1))

    if res_mean is None:
        res_mean = 0

    if res_var is None:
        res_var = 1

    if Q is None:
        Q = numpy.eye(npts)

    VarMean = res_var * X.T.dot(Q).dot(X) / npts**2

    # T  =  mean(y,0)/s-hat-2
    # use diag to get s_hat2 for each variable
    T = (numpy.mean(y, 0)-res_mean
         )/numpy.sqrt(VarMean)*numpy.sqrt(res_var) + res_mean

    # Assuming variance is estimated on whole image
    # and assuming infinite df
    p = 1 - scipy.stats.norm.cdf(T)

    return(T, p)


def run_ttests(narps, logfile,
               overwrite=True):
    masker = nilearn.input_data.NiftiMasker(
        mask_img=narps.dirs.MNI_mask)
    results_dir = narps.dirs.dirs['consensus']

    func_name = sys._getframe().f_code.co_name
    log_to_file(
        logfile, '%s' %
        func_name)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    for hyp in hypnums:
        if not overwrite and os.path.exists(os.path.join(
                results_dir,
                'hypo%d_1-fdr.nii.gz' % hyp)):
            print('using existing results')
            continue
        print('running consensus analysis for hypothesis', hyp)
        maps = glob.glob(os.path.join(
            narps.dirs.dirs['output'],
            'zstat/*/hypo%d_unthresh.nii.gz' % hyp))
        maps.sort()
        data = masker.fit_transform(maps)

        # get estimated mean, variance, and correlation for t_corr
        img_mean = numpy.mean(data)
        img_var = numpy.mean(numpy.var(data, 1))
        cc = numpy.corrcoef(data)
        log_to_file(
            logfile,
            'mean = %f, var = %f, mean_cc = %f' %
            (img_mean, img_var,
            numpy.mean(cc[numpy.triu_indices_from(cc, 1)])))

        # perform t-test
        tvals, pvals = t_corr(data,
                              res_mean=img_mean,
                              res_var=img_var,
                              Q=cc)

        # move back into image format
        timg = masker.inverse_transform(tvals)
        timg.to_filename(os.path.join(results_dir, 'hypo%d_t.nii.gz' % hyp))
        pimg = masker.inverse_transform(1-pvals)
        pimg.to_filename(os.path.join(results_dir, 'hypo%d_1-p.nii.gz' % hyp))
        fdr_results = multipletests(pvals[0, :], 0.05, 'fdr_tsbh')
        log_to_file(
            logfile,
            "%d voxels significant at FDR corrected p<.05" %
            numpy.sum(fdr_results[0]))
        fdrimg = masker.inverse_transform(1 - fdr_results[1])
        fdrimg.to_filename(os.path.join(
            results_dir,
            'hypo%d_1-fdr.nii.gz' % hyp))


def mk_figures(narps, logfile, thresh=0.95):

    func_name = sys._getframe().f_code.co_name
    log_to_file(
        logfile, '%s' %
        func_name)

    fig, ax = plt.subplots(7, 1, figsize=(12, 24))
    cut_coords = [-24, -10, 4, 18, 32, 52, 64]

    for i, hyp in enumerate(hypnums):
        pmap = os.path.join(
            narps.dirs.dirs['consensus'],
            'hypo%d_1-fdr.nii.gz' % hyp)
        tmap = os.path.join(
            narps.dirs.dirs['consensus'],
            'hypo%d_t.nii.gz' % hyp)
        pimg = nibabel.load(pmap)
        timg = nibabel.load(tmap)
        pdata = pimg.get_fdata()
        tdata = timg.get_fdata()[:, :, :, 0]
        threshdata = (pdata > thresh)*tdata
        threshimg = nibabel.Nifti1Image(threshdata, affine=timg.affine)
        nilearn.plotting.plot_stat_map(
            threshimg,
            threshold=0.1,
            display_mode="z",
            colorbar=True,
            title='hyp %d:' % hyp+hypotheses[hyp],
            vmax=8,
            cmap='jet',
            cut_coords=cut_coords,
            axes=ax[i])

    plt.savefig(os.path.join(
        narps.dirs.dirs['figures'],
        'consensus_map.pdf'))
    plt.close(fig)


if __name__ == "__main__":
    # set an environment variable called NARPS_BASEDIR
    # with location of base directory
    if 'NARPS_BASEDIR' in os.environ:
        basedir = os.environ['NARPS_BASEDIR']
    else:
        basedir = '/data'

    # setup main class
    narps = Narps(basedir)
    narps.load_data()
    narps.dirs.dirs['consensus'] = os.path.join(
        narps.dirs.dirs['output'],
        'consensus_analysis')

    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        '%s.txt' % sys.argv[0].split('.')[0])
    log_to_file(
        logfile, 'running %s' %
        sys.argv[0].split('.')[0],
        flush=True)

    if not os.path.exists(narps.dirs.dirs['consensus']):
        os.mkdir(narps.dirs.dirs['consensus'])

    run_ttests(narps, logfile)
    mk_figures(narps, logfile)
