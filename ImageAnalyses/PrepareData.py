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
from neurovault_collection_downloader import cli
from utils import log_to_file

# these are teams that used surface-based analysis so must be excluded
# from map analyses
TEAMS_TO_SKIP = ['1K0E', 'X1Z4']


def get_download_dir(basedir, overwrite=True):
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
        if not isinstance(teaminfo.loc[t, 'public link'], str):
            public_link = None
        else:
            public_link = teaminfo.loc[t, 'public link']
            public_link = fix_trailing_slashes(public_link)

        private_link = teaminfo.loc[t, 'private link']
        private_link = fix_trailing_slashes(private_link)
        if public_link is not None:
            collectionID[teamID] = os.path.basename(public_link)
        else:
            collectionID[teamID] = os.path.basename(private_link)
        if verbose:
            print(teamID, collectionID[teamID], private_link, public_link)
        assert len(collectionID[teamID]) > 3

    return(collectionID)


def download_collections(collectionIDs, download_dir,
                         verbose=True, overwrite=True):
    teamIDs = list(collectionIDs.keys())
    teamIDs.sort()  # to maintain order
    failed_downloads = []
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
                    os.path.join(download_dir, teamID))

                # clean out subject files
                subfiles = glob.glob(os.path.join(
                    download_dir,
                    teamID,
                    'sub*'))
                for s in subfiles:
                    os.remove(s)
            else:
                if verbose:
                    print('using existing data for', teamID)
        except:  # noqa - need to catch unknown exceptions here
            if verbose:
                print('download failed for', teamID)
            failed_downloads.append(teamID)
    return(failed_downloads)


def check_downloads(completed_downloads):
    """
    check for complete downloads
    """

    missing_files = {}
    for c in completed_downloads:
        datadir = glob.glob(os.path.join(c, '*'))[0]
        teamID = os.path.basename(c)
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
    for root, dirs, files in os.walk(download_dir):
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


def fix_names(collectionIDs, download_dir):
    pass


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description='Process NARPS data')
    parser.add_argument('-b', '--basedir',
                        help='base directory')
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='use testing mode (no processing)')
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

    completed_downloads = glob.glob(os.path.join(download_dir, '*'))
    print('found %d completed downloads' % len(completed_downloads))

    missing_files = check_downloads(completed_downloads)

    has_missing_files = [
        teamID for teamID in missing_files
        if len(missing_files[teamID]) > 0]
    print('found %d teams with missing/misnamed files' % len(
        has_missing_files))

    # get manifest and hashes
    logfile = os.path.join(
        basedir,
        'logs/MANIFEST.neurovault')

    log_to_file(
        logfile,
        'Getting data from neurovault',
        flush=True)
    log_data(
        download_dir,
        logfile)

    # fix_names(collectionIDs, download_dir)
