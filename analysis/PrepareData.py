"""
obtain the data from neurovault and reformat as needed
"""

import os
import glob
import shutil
import pandas
from neurovault_collection_downloader import cli

# these are teams that used surface-based analysis so must be excluded
# from map analyses
TEAMS_TO_SKIP = ['1K0E', 'X1Z4']


def get_download_dir(basedir, use_existing=False):
    download_dir = os.path.join(basedir, 'neurovault_downloads')
    if not use_existing:
        if os.path.exists(download_dir):
            print('removing existing downloads directory')
            shutil.rmtree(download_dir)

    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    return(download_dir)


def get_collection_ids(infile='collection_teamID_updated.xlsx',
                       verbose=False):
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
            public_link = teaminfo.loc[t, 'public link'].strip('/')

        private_link = teaminfo.loc[t, 'private link'].strip('/')
        if public_link is not None:
            collectionID[teamID] = os.path.basename(public_link)
        else:
            collectionID[teamID] = os.path.basename(private_link)
        if verbose:
            print(teamID, collectionID[teamID], private_link, public_link)
        assert len(collectionID[teamID]) > 3

    return(collectionID)


def download_collections(collectionIDs, download_dir,
                         use_existing=False, verbose=True):
    teamIDs = list(collectionIDs.keys())
    teamIDs.sort()  # to maintain order
    failed_downloads = []
    for teamID in teamIDs:
        if teamID in TEAMS_TO_SKIP:
            if verbose:
                print('skipping', teamID)
            continue
        try:
            if not use_existing or \
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


def fix_names(collectionIDs, download_dir):
    pass


if __name__ == "__main__":
    if 'NARPS_BASEDIR' in os.environ:
        basedir = os.environ['NARPS_BASEDIR']
    else:
        basedir = '/data'

    use_existing = True
    skip_download = True

    download_dir = get_download_dir(basedir, use_existing)
    print('downloading data to', basedir)

    collectionIDs = get_collection_ids()
    print('found', len(collectionIDs), 'collections')

    if not skip_download:
        failed_downloads = download_collections(
            collectionIDs,
            download_dir,
            use_existing=use_existing)

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

    fix_names(collectionIDs, download_dir)
