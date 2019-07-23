"""
compare results from simulated analysis
to actual consensus results
"""

import os
import numpy
import nilearn.input_data
from narps import Narps, NarpsDirs, hypnums # noqa, flake8 error
from utils import log_to_file

thresh = 0.95
basedir = os.environ['NARPS_BASEDIR']
simdir = basedir + '_simulated'
print(basedir)
assert os.path.exists(simdir)

narps = Narps(basedir)

masker = nilearn.input_data.NiftiMasker(mask_img=narps.dirs.MNI_mask)
cc = {}
for hyp in hypnums:
    consensus_imgfile = os.path.join(
        basedir,
        'output/consensus_analysis/hypo%d_t.nii.gz' % hyp)
    consensus_imgdata = masker.fit_transform(consensus_imgfile)
    simdata_imgfile = os.path.join(
        simdir,
        'output/consensus_analysis/hypo%d_t.nii.gz' % hyp)
    simdata_imgdata = masker.fit_transform(simdata_imgfile)
    cc[hyp] = numpy.corrcoef(
        simdata_imgdata,
        consensus_imgdata)[0, 1]


try:
    assert numpy.array(list(cc.values())).min() > thresh
    log_to_file(
        os.path.join(simdir, 'logs/simulated_data.log'),
        'SUCESSS: all hypotheses correlated > %0.2f' % thresh)
except AssertionError:
    log_to_file(
        os.path.join(simdir, 'logs/simulated_data.log'),
        'FAILURE')
    print(cc)
