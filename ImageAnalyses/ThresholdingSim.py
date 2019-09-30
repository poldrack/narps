#!/usr/bin/env python
# coding: utf-8
"""
simulate effects of a single thresholding method across all data
using unthresholded Z maps

H/O regions that overlap vmpfc: 27, 28, 29, 25, 1
also use neurosynth map for "ventromedial frontal"

From Rotem:
1) see the same analysis with the striatum (since these unthresh images are for hypo1 and 3). Let's see if it is specifically the vmpfc.
2) Run this new analysis only for the teams in cluster1 for hypo1 (we saw their unthresh maps were very similar while decisions varied considerably).
3) test similarity of unthresh maps for hypo1 only in grey matter / with an approach somewhat similar to the unthresh, e.g. take only voxels that exceed a specific threshold before computing the similarity.
4) Run through a 4d image with the unthresh maps of all teams for hypo1, and the same for the thresh maps created by this new analysis. Let's eyeball to see how it looks like.

"""

import os
import argparse
import numpy
import glob
import pandas
import nilearn.input_data
from get_3d_peaks import get_3d_peaks
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
            narps.dirs.dirs['orig'],
            'neurosynth/ventromedialprefrontal_association-test_z_FDR_0.01.nii.gz'
        )
    else:
        neurosynth_img = None

    masker = nilearn.input_data.NiftiMasker(MNI_img)
    HO_data = masker.fit_transform(HO_img)[0,:]
    
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
        narps.dirs.dirs['zstat'], '*'
        ))
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
        z = masker.fit_transform(img)[0]
        if simulate_noise:
            z = numpy.random.randn(z.shape[0])
        p = 1 - scipy.stats.norm.cdf(z)
        # compute fdr across whole brain
        fdr_results = multipletests(
            p, fdr, 'fdr_tsbh')
        if numpy.sum(fdr_results[0]) > 0:
            fdr_thresh[i, 0] = numpy.max(
                p[fdr_results[0]==True])
        else:
            fdr_thresh[i, 0] = numpy.nan

        # compute fdr only on ROI voxels
        p_roi = p[roi_mask > 0]
        fdr_results_roi = multipletests(
            p_roi, fdr, 'fdr_tsbh')
        if numpy.sum(fdr_results_roi[0]) > 0:
            fdr_thresh[i, 1] = numpy.max(
                p_roi[fdr_results_roi[0]==True])
        else:
            fdr_thresh[i, 1] = numpy.nan

    return(numpy.nanmean(fdr_thresh,0))

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
    roi_mask = masker.fit_transform(maskimg)[0,:]

    mean_fdr_thresh = get_mean_fdr_thresh(
            zstat_imgs, masker, roi_mask,
            simulate_noise)

    results = pandas.DataFrame({'Uncorrected': numpy.zeros(len(zstat_imgs))})
    results['FDR'] = 0.0
    results['FDR (only within ROI)'] = 0.0
    for i, img in enumerate(zstat_imgs):
        z = masker.fit_transform(img)[0]
        if simulate_noise:
            z = numpy.random.randn(z.shape[0])
        p = 1 - scipy.stats.norm.cdf(z)
        results.iloc[i, 0] = numpy.sum(p[roi_mask > 0] < pthresh)
        results.iloc[i, 1] = numpy.sum(p[roi_mask > 0] < mean_fdr_thresh[0])
        results.iloc[i, 2] = numpy.sum(p[roi_mask > 0] < mean_fdr_thresh[1])
    
    message = '\nHypothesis: %s\n' % hyp
    if simulate_noise:
        message += 'SIMULATING WITH RANDOM NOISE\n'
    message += 'Region: %s\n' % region
    message += 'ROI image: %s\n' % maskimg_file
    message += 'Using mean FDR thresholds (whole brain/roi): %s\n' %\
         mean_fdr_thresh
    message += '\nProportion with activation:\n'
    message += (results>0).mean(0).to_string() + '\n'
    log_to_file(logfile, message)
    return(results)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Thresholding simulation')
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
    narps.dirs.get_output_dir('ThresholdSimulation',
                              base='figures')

    logfile = os.path.join(
        narps.dirs.dirs['logs'],
        'ThresholdSimulation.log')

    if not args.test:

        log_to_file(
            logfile,
            'Running thresholding simulation',
            flush=True)

        for hyp in range(1, 10):
            activations = get_activations(
                narps, hyp, logfile, simulate_noise=False)

