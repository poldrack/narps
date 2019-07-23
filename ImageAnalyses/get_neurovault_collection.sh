#!/bin/bash
# USAGE: get_neurovault_collection.sh <collectionID> <outputDir>
# download a dataset from neurovault
# using https://github.com/NeuroVault/neurovault_collection_downloader


echo $1 >| /tmp/collections.txt
neurovault_collection_downloader /tmp/collections.txt $2
rm /tmp/collections.txt