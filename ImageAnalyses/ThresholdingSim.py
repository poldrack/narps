#!/usr/bin/env python
# coding: utf-8
"""
simulate effects of a single thresholding method across all data
using unthresholded Z maps

H/O regions that overlap vmpfc: 27, 28, 29, 25, 1
also use neurosynth map for "ventromedial frontal"

"""

import os
import argparse
import numpy
import glob
import pandas
import nilearn.input_data
import nibabel
from statsmodels.stats.multitest import multipletests
import scipy.stats

from utils import log_to_file
from narps import Narps

cut_coords = [-24, -10, 4, 18, 32, 52, 64]
bins = numpy.linspace(-5, 5)

hypnums = [i for i in range(1, 10)]

# H-O region values
region_rois = {'vmpfc': [27, 28, 29, 25, 1],
               'ventralstriatum': [11, 21],
               'amygdala': [10, 20]}
hyp_regions = {
    1: 'vmpfc',
    2: 'vmpfc',
    3: 'ventralstriatum',
    4: 'ventralstriatum',
    5: 'vmpfc',
    6: 'vmpfc',
    7: 'amygdala',
    8: 'amygdala',
    9: 'amygdala'
}


def get_mask_img(narps, region):
    assert region in region_rois
    maskimg = os.path.join(
            narps.dirs.dirs['ThresholdSimulation'],
            '%s_mask.nii.gz' % region
        )
    if not os.path.exists(maskimg):
        create_mask_img(narps, region)
    return(maskimg)


def create_mask_img(narps, region):
    maskimg = os.path.join(
        narps.dirs.dirs['ThresholdSimulation'],
        '%s_mask.nii.gz' % region
    )
    if region in ['ventralstriatum', 'amygdala']:
        HO_base = 'HarvardOxford-sub-maxprob-thr25-2mm.nii.gz'
    else:
        HO_base = 'HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'

    HO_img = os.path.join(
        os.environ['FSLDIR'],
        'data/atlases/HarvardOxford', HO_base
    )

    MNI_img = os.path.join(
        os.environ['FSLDIR'],
        'data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    )

    if region == 'vmpfc':
        neurosynth_img = os.path.join(
            narps.dirs.dirs['orig'], 'neurosynth',
            'ventromedialprefrontal_association-test_z_FDR_0.01.nii.gz'
        )
    else:
        neurosynth_img = None

    masker = nilearn.input_data.NiftiMasker(MNI_img)
    HO_data = masker.fit_transform(HO_img)[0, :]

    # find voxels in any of the HO regions overlapping with VMPFC
    matches = None
    for roi in region_rois[region]:
        if matches is None:
            matches = numpy.where(HO_data == roi)[0]
        else:
            matches = numpy.hstack((matches, numpy.where(HO_data == roi)[0]))
    HO_region_mask = numpy.zeros(HO_data.shape[0])
    HO_region_mask[matches] = 1

    # intersect with neurosynth map
    if neurosynth_img is not None:
        neurosynth_data = masker.fit_transform(neurosynth_img)[0]
        neurosynth_mask = (neurosynth_data > 1e-8).astype('int')
        combo_mask = neurosynth_mask*HO_region_mask
    else:
        combo_mask = HO_region_mask

    combo_mask_img = masker.inverse_transform(combo_mask)
    combo_mask_img.to_filename(maskimg)


def get_zstat_images(narps, hyp):
    imgfiles = []
    zdirs = glob.glob(os.path.join(
        narps.dirs.dirs['zstat'], '*'))
    for d in zdirs:
        imgfile = os.path.join(
            d, 'hypo%d_unthresh.nii.gz' % hyp)
        if os.path.exists(imgfile):
            imgfiles.append(imgfile)
    return(imgfiles)


def get_mean_fdr_thresh(zstat_imgs, masker,
                        roi_mask, simulate_noise,
                        fdr=0.05):
    # get average thresh for whole brain and ROI
    fdr_thresh = numpy.zeros((len(zstat_imgs), 2))
    for i, img in enumerate(zstat_imgs):
        z = masker.fit_transform(img)[0, :]
        if simulate_noise:
            z = numpy.random.randn(z.shape[0])
        p = 1 - scipy.stats.norm.cdf(z)
        # compute fdr across whole brain
        fdr_results = multipletests(
            p, fdr, 'fdr_tsbh')
        if numpy.sum(fdr_results[0]) > 0:
            fdr_thresh[i, 0] = numpy.max(
                p[fdr_results[0]])
        else:
            # use Bonferroni if there are no
            # suprathreshold voxels
            fdr_thresh[i, 0] = fdr/len(p)

        # compute fdr only on ROI voxels
        p_roi = p[roi_mask > 0]
        fdr_results_roi = multipletests(
            p_roi, fdr, 'fdr_tsbh')
        if numpy.sum(fdr_results_roi[0]) > 0:
            fdr_thresh[i, 1] = numpy.max(
                p_roi[fdr_results_roi[0]])
        else:
            # use Bonferroni if there are no
            # suprathreshold voxels
            fdr_thresh[i, 1] = fdr/len(p_roi)

    return(numpy.mean(fdr_thresh, 0))


def get_activations(narps, hyp, logfile,
                    fdr=0.05, pthresh=0.001,
                    simulate_noise=False):

    assert fdr is not None or pthresh is not None

    region = hyp_regions[hyp]

    # load mask, create if necessary
    maskimg_file = get_mask_img(narps, region)
    maskimg = nibabel.load(maskimg_file)

    # load data
    zstat_imgs = get_zstat_images(narps, hyp)

    # setup masker
    MNI_img = os.path.join(
        os.environ['FSLDIR'],
        'data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    )
    masker = nilearn.input_data.NiftiMasker(MNI_img)

    # get roi mask
    roi_mask = masker.fit_transform(maskimg)[0, :]

    mean_fdr_thresh = get_mean_fdr_thresh(
            zstat_imgs, masker, roi_mask,
            simulate_noise)

    results = pandas.DataFrame({'Uncorrected': numpy.zeros(len(zstat_imgs))})
    results['FDR'] = 0.0
    results['FDR (only within ROI)'] = 0.0
    for i, img in enumerate(zstat_imgs):
        z = masker.fit_transform(img)[0, :]
        if simulate_noise:
            z = numpy.random.randn(z.shape[0])
        p = 1 - scipy.stats.norm.cdf(z)  # convert Z to p
        # count how many voxels in ROI mask are below each threshold
        results.iloc[i, 0] = numpy.sum(p[roi_mask > 0] < pthresh)
        results.iloc[i, 1] = numpy.sum(p[roi_mask > 0] < mean_fdr_thresh[0])
        results.iloc[i, 2] = numpy.sum(p[roi_mask > 0] < mean_fdr_thresh[1])

    # load ALE and consensus results for comparison
    meta_results = numpy.zeros(2)
    ale_img = os.path.join(
        narps.dirs.dirs['output'],
        'ALE/hypo%d_fdr_thresholded.nii.gz' % hyp)
    if os.path.exists(ale_img):
        ale_data = masker.fit_transform(ale_img)[0, :]
        meta_results[0] = numpy.sum(ale_data[roi_mask > 0])
    else:
        meta_results[0] = numpy.nan
    # consensus not performed for 3 and 4, so use 1/2 instead
    hyp_fix = {1: 1,
               2: 2,
               3: 1,
               4: 2,
               5: 5,
               6: 6,
               7: 7,
               8: 8,
               9: 9}
    consensus_img = os.path.join(
        narps.dirs.dirs['output'],
        'consensus_analysis/hypo%d_1-fdr.nii.gz' % hyp_fix[hyp])
    if os.path.exists(ale_img):
        consensus_data = masker.fit_transform(consensus_img)[0, :]
        meta_results[1] = numpy.sum(
            consensus_data[roi_mask > 0] > (1 - fdr))
    else:
        meta_results[1] = numpy.nan

    message = '\nHypothesis: %s\n' % hyp
    if simulate_noise:
        message += 'SIMULATING WITH RANDOM NOISE\n'
    message += 'Region (%d voxels): %s\n' % (
        numpy.sum(roi_mask), region)
    message += 'ROI image: %s\n' % maskimg_file
    message += 'Using mean FDR thresholds (whole brain/roi): %s\n' %\
        mean_fdr_thresh
    message += '\nProportion with activation:\n'
    message += (results > 0).mean(0).to_string() + '\n'
    message += 'Activated voxels in ALE map: %d\n' % meta_results[0]
    message += 'Activated voxels in consensus map: %d\n' % meta_results[1]
    log_to_file(logfile, message)
    return(results, mean_fdr_thresh, meta_results,
           numpy.sum(roi_mask))


def run_all_analyses(narps, simulate_noise=False):
    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'ThresholdSimulation.log')
    log_to_file(
        logfile,
        'Running thresholding simulation',
        flush=True)

    all_results = []
    for hyp in range(1, 10):
        results, mean_fdr_thresh, meta_results, roisize = get_activations(
            narps, hyp, logfile,
            simulate_noise=simulate_noise)
        mean_results = (results > 0).mean(0)
        r = [hyp, roisize,
             mean_results[0],
             mean_results[1],
             mean_fdr_thresh[0],
             mean_results[2],
             mean_fdr_thresh[1],
             meta_results[0],
             meta_results[1]]
        all_results.append(r)

    results_df = pandas.DataFrame(all_results, columns=[
        'Hypothesis',
        'N voxels in ROI',
        'p(Uncorrected)',
        'p(whole-brain FDR)',
        'FDR cutoff (whole-brain)',
        'p(SVC FDR)',
        'FDR cutoff (SVC)',
        'ALE (n voxels in ROI)',
        'Consensus (n voxels in ROI)'])
    results_df.to_csv(os.path.join(
        narps.dirs.dirs['ThresholdSimulation'],
        'simulation_results.csv'
    ))
    # compare with ALE

    return(results_df)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Thresholding simulation')
    parser.add_argument('-b', '--basedir',
                        help='base directory')
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='use testing mode (no processing)')
    parser.add_argument('-s', '--simulate_noise',
                        action='store_true',
                        help='test using random noise')
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
    narps.dirs.get_output_dir('ThresholdSimulation',
                              base='figures')

    if not args.test:
        all_results = run_all_analyses(
            narps, args.simulate_noise)
