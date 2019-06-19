# download updated datasets
# using https://github.com/NeuroVault/neurovault_collection_downloader

# VG39 - BPZDIIWY
# L3V8- CMJIFMMR


# echo """CMJIFMMR""" > collections.txt
# neurovault_collection_downloader collections.txt /Users/poldrack/data_unsynced/NARPS/maps/updates
# mv /Users/poldrack/data_unsynced/NARPS/maps/updates/CMJIFMMR /Users/poldrack/data_unsynced/NARPS/maps/updates/CMJIFMMR_L3V8

#echo """BPZDIIWY""" > collections.txt
#neurovault_collection_downloader collections.txt /Users/poldrack/data_unsynced/NARPS/maps/updates

#mv /Users/poldrack/data_unsynced/NARPS/maps/updates/BPZDIIWY /Users/poldrack/data_unsynced/NARPS/maps/updates/BPZDIIWY_VG39

echo """TMZGNUWI""" > collections.txt
neurovault_collection_downloader collections.txt /Users/poldrack/data_unsynced/NARPS/maps/updates

mv /Users/poldrack/data_unsynced/NARPS/maps/updates/TMZGNUWI /Users/poldrack/data_unsynced/NARPS/maps/updates/TMZGNUWI_R5K7