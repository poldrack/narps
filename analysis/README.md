### Code for NARPS data analysis

#### Data preparation

The code for data preparation is contained in narps.py, with some utility functions in utils.py.

To run the full preparation and generate some diagnostic figures, execute PrepareMaps.ipynb.  This requires that the raw data directories for all teams are placed in a directory called *maps/orig* within the main data directory (which you should set as the base directory in the notebook).  In addition, these files are required within the base directory:

Metadata files:

- metadata/analysis_pipelines_SW.xlsx
- metadata/narps_neurovault_images_details.csv
- metadata/narps_results.xlsx

MNI templates from FSL distribution:

- maps/templates/MNI152_T1_2mm.nii.gz
- maps/templates/MNI152_T1_2mm_brain_mask.nii.gz


#### Metadata preparation

Once the data preparation has been completed, generation of the main metadata file should be performed by executing PrepareMetadata.ipynb

#### Map analysis

One metadata have been prepared, summary of the maps can be run using AnalyzeMaps.ipynb

#### Decision analysis

Analysis of the reported decisions can be run using DecisionAnalysis.Rmd.