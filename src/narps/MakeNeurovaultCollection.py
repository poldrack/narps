"""
create neurovault collection with results
"""

import os
import glob
from pynv import Client

assert 'NEUROVAULT_API_KEY' in os.environ
access_token = os.environ['NEUROVAULT_API_KEY']

assert 'NARPS_BASEDIR' in os.environ
basedir = os.environ['NARPS_BASEDIR']
output_dir = os.path.join(basedir, 'output')

api = Client(access_token=access_token)


# tuple contains directory name and wildcard
imagesets = {
    'overlap': (
        'overlap_binarized_thresh',
        'hypo*.nii.gz',
        'Other',
        'NARPS Study: Overlap maps, showing roportion of teams with significant activity at any voxel (Figure 1)'), # noqa
    'clustermaps': (
        'cluster_maps',
        'hyp*_cluster*_mean.nii.gz',
        'Other',
        'NARPS Study: Cluster maps, showing mean Z statistic across teams for each hypothesis/cluster (Figure 2, Supplementary Figures 2-7)'), # noqa
    'CBMA': (
        'ALE',
        'hypo*_fdr_oneminusp.nii.gz',
        'Other',
        'NARPS Study: Coordinate-based meta-analysis results (1 - P after FDR correction) (Supplementary Figure 1)'), # noqa
    'tau': (
        'consensus_analysis',
        'hypo*_tau.nii.gz',
        'Other',
        'NARPS Study: Maps of estimated between-team variability (tau) at each voxel (Supplementary Figure 8)'), # noqa
    'IBMA': (
        'consensus_analysis',
        'hypo*_1-fdr.nii.gz',
        'Other',
        'NARPS Study: Image-based meta-analysis results (1 - P after FDR correction) (Supplementary Figure 10)')} # noqa

upload = True
make_private = True


def find_collection(api, **kwargs):
    my_collections = api.my_collections()['results']
    assert len(kwargs) == 1
    k = list(kwargs.keys())[0]
    matches = []
    for c in my_collections:
        if k in c:
            if c[k] == kwargs[k]:
                matches.append(c['id'])
    return(matches)


collection = {}
for imgset in imagesets:
    if upload and imgset not in collection:
        cname = 'NARPS_%s' % imgset
        c = find_collection(api, name=cname)
        if len(c) > 0:
            collection[imgset] = api.get_collection(c[0])
        else:
            collection[imgset] = api.create_collection(
                cname,
                description=imagesets[imgset][3])

    imgs = glob.glob(
        os.path.join(
            output_dir,
            imagesets[imgset][0],
            imagesets[imgset][1]))
    imgs.sort()
    for img in imgs:
        # get hypothesis number
        hypnum = int([
            i for i in os.path.basename(img).split('_')
            if i.find('hyp') > -1][0].split('.')[0].replace(
                'hypo', 'hyp').replace('hyp', ''))
        print(imgset, hypnum)
        if upload:
            image = api.add_image(
                collection[imgset]['id'],
                img,
                name='%s_Hyp%d' % (imgset, hypnum),
                modality='fMRI-BOLD',
                map_type=imagesets[imgset][2],
                target_template_image='GenericMNI'
            )
