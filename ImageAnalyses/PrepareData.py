"""
obtain the data from neurovault and reformat as needed
for sharing
"""

import os
import hashlib
import argparse
import glob
import shutil
import pandas
import json

from neurovault_collection_downloader import cli
from utils import log_to_file

# these are teams that are excluded
# from map analyses:
# used surface-based analysis ('1K0E', 'X1Z4')
# badly registered ( 'L1A8')
# used SVC analysis which was not allowed ('VG39')
TEAMS_TO_SKIP = ['1K0E', 'X1Z4', 'L1A8', 'VG39']

# incorrect unthresh values (very small) (569K)
# did not report t/z stats (16IN)

TEAMS_TO_REMOVE_UNTHRESH = ['569K', '16IN']


def get_download_dir(basedir, overwrite=False):
    download_dir = os.path.join(basedir, 'neurovault_downloads')
    if overwrite:
        if os.path.exists(download_dir):
            print('removing existing downloads directory')
            shutil.rmtree(download_dir)

    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    return(download_dir)


def fix_trailing_slashes(s):
    """ remove abritrary number of trailing slashes"""
    s = s.strip()  # first remove spaces
    while s[-1] == '/':
        s = s.strip('/')
    return(s)


def get_collection_ids(infile='collection_teamID_updated.xlsx',
                       verbose=True):
    teaminfo = pandas.read_excel(infile)
    collectionID = {}
    for t in teaminfo.index:
        teamID = teaminfo.loc[t, 'team ID']
        if teamID in TEAMS_TO_SKIP:
            if verbose:
                print('skipping', teamID)
            continue

        public_link = teaminfo.loc[t, 'new NV link']
        public_link = fix_trailing_slashes(public_link)

        collectionID[teamID] = os.path.basename(public_link)
        if verbose:
            print(teamID, collectionID[teamID], public_link)
        assert len(collectionID[teamID]) > 3

    return(collectionID)


def download_collections(collectionIDs, download_dir,
                         verbose=True, overwrite=False):
    teamIDs = list(collectionIDs.keys())
    teamIDs.sort()  # to maintain order
    failed_downloads = {}
    for teamID in teamIDs:
        if teamID in TEAMS_TO_SKIP:
            if verbose:
                print('skipping', teamID)
            continue
        try:
            if overwrite or \
                    not os.path.exists(os.path.join(download_dir, teamID)):
                if verbose:
                    print('fetching', teamID)
                cli.fetch_collection(
                    collectionIDs[teamID],
                    download_dir,
                    '%s_%s' % (collectionIDs[teamID], teamID),
                    exclude_tag=['sub', 'cope'])

            else:
                if verbose:
                    print('using existing data for', teamID)
        except Exception as e:  # noqa - need to catch unknown exceptions here
            print('download failed for', teamID)
            print(e)
            failed_downloads[teamID] = e

    return(failed_downloads)


def check_downloads(completed_downloads):
    """
    check for complete downloads
    """

    missing_files = {}
    for datadir in completed_downloads:
        teamID = os.path.basename(datadir)
        files = glob.glob(os.path.join(datadir, '*'))
        print('%s: found %d files' % (teamID, len(files)))
        # check for necessary files:
        filenames = [os.path.basename(f).lower() for f in files]
        missing_files[teamID] = []

        for hyp in range(1, 10):
            for imgtype in ['thresh', 'unthresh']:
                targfile = 'hypo%d_%s.nii.gz' % (hyp, imgtype)
                if targfile not in filenames:
                    missing_files[teamID].append(targfile)
        if len(missing_files[teamID]) > 0:
            print('missing %d files' % len(missing_files[teamID]))
            print(filenames)
    return(missing_files)


def log_data(download_dir,
             logfile, verbose=True):
    """record manifest and file hashes"""
    imgfiles = {}
    # traverse root directory, and list directories as dirs and files as files
    for root, _, files in os.walk(download_dir):
        path = root.split(os.sep)
        for file in files:
            if file.find('.nii.gz') < 0:
                # skip non-nifti files
                continue
            fname = os.path.join(root, file)
            filehash = hashlib.md5(open(fname, 'rb').read()).hexdigest()
            short_fname = os.path.join('/'.join(path[-2:]), file)
            imgfiles[short_fname] = filehash
            if verbose:
                print(short_fname, filehash)
            log_to_file(
                logfile,
                '%s %s' % (short_fname, filehash))


def copy_renamed_files(collectionIDs, download_dir, logfile):
    """change file names based on info in images.json"""
    # setup target directory
    orig_dir = os.path.join(
        os.path.dirname(download_dir),
        'orig'
    )
    if not os.path.exists(orig_dir):
        os.mkdir(orig_dir)

    for teamID in collectionIDs:
        collectionID = '%s_%s' % (
                collectionIDs[teamID],
                teamID)
        collection_dir = os.path.join(
            download_dir,
            collectionID)
        fixed_dir = os.path.join(
            orig_dir,
            collectionID)
        if not os.path.exists(fixed_dir):
            os.mkdir(fixed_dir)

        jsonfile = os.path.join(
            collection_dir,
            'images.json')
        if not os.path.exists(jsonfile):
            print('no json file for ', collectionID)
            continue
        with open(jsonfile) as f:
            image_info = json.load(f)
        for img in image_info:
            origname = os.path.basename(img['file'])
            # fix various issues with names
            newname = img['name'].replace(
                'tresh', 'thresh').replace(' ', '_')+'.nii.gz'
            newname = newname.replace(
                'hypo_', 'hypo').replace(
                    'uthresh', 'unthresh').replace(
                        '_LR', ''
                    )

            # skip unthresh images if necessary
            if newname.find('unthresh') > -1 and \
                    teamID in TEAMS_TO_REMOVE_UNTHRESH:
                continue

            if origname.find('sub') > -1 or \
                    not newname.find('thresh') > -1:  # skip sub files
                continue
            else:
                log_to_file(
                    logfile,
                    'copying %s/%s to %s/%s' % (
                        collectionID, origname, collectionID, newname))
                shutil.copy(
                    os.path.join(collection_dir, origname),
                    os.path.join(fixed_dir, newname))
    return(orig_dir)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description='Process NARPS data')
    parser.add_argument('-b', '--basedir',
                        help='base directory')
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='use testing mode (no processing)')
    parser.add_argument('-l', '--leave_downloads',
                        action='store_true',
                        help='do not delete downloads')
    parser.add_argument('-s', '--skip_download',
                        action='store_true',
                        help='use existing data')
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

    if not os.path.exists(basedir):
        os.mkdir(basedir)

    # set up logging
    logfile = os.path.join(
        basedir,
        'logs/neurovault_download.log')
    if not os.path.exists(os.path.dirname(logfile)):
        os.mkdir(os.path.dirname(logfile))

    log_to_file(
        logfile,
        'Getting data from neurovault',
        flush=True)

    collectionIDs = get_collection_ids()
    print('found', len(collectionIDs), 'collections')

    if args.skip_download:
        download_dir = get_download_dir(basedir, overwrite=False)
        assert os.path.exists(download_dir)
    else:
        download_dir = get_download_dir(basedir)
        print('downloading data to', basedir)

        failed_downloads = download_collections(
            collectionIDs,
            download_dir)

        if len(failed_downloads) > 0:
            print('failed downloads for %d teams' % len(failed_downloads))
            print(failed_downloads)
            for f in failed_downloads:
                log_to_file(
                    logfile,
                    '%s: %s' % (f, failed_downloads[f]))

    renaming_logfile = os.path.join(
        basedir,
        'logs/neurovault_renaming.log')
    orig_dir = copy_renamed_files(
        collectionIDs,
        download_dir,
        renaming_logfile)

    completed_downloads = [
        i for i in glob.glob(os.path.join(orig_dir, '*'))
        if os.path.isdir(i)]
    print('found %d completed downloads' % len(completed_downloads))

    missing_files = check_downloads(completed_downloads)

    has_missing_files = [
        teamID for teamID in missing_files
        if len(missing_files[teamID]) > 0]
    log_to_file(
        logfile,
        'found %d teams with missing/misnamed files:' % len(
            has_missing_files))
    log_to_file(
        logfile,
        ' '.join(has_missing_files))

    if not os.path.exists(os.path.join(basedir, 'logs')):
        os.mkdir(os.path.join(basedir, 'logs'))

    manifest_file = os.path.join(
        basedir,
        'logs/MANIFEST.neurovault')
    log_data(
        download_dir,
        manifest_file)

    if not args.leave_downloads:
        shutil.rmtree(download_dir)
