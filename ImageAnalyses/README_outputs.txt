Outputs from automated analysis code for NARPS project
See https://github.com/poldrack/narps for more details

Legend for files contained in this directory:


./image_diagnostics_zstat: image diagnostics for resampled statistical images (one file per team)

./image_diagnostics_orig: image diagnostics for original (submitted) statistical images (one file per team)

./output: outputs from analyses

./output/npboot_output.RData: stored samples from nonparametric bootstrap analysis

./output/overlap_binarized_thresh: images showing proportion of teams with activation at each voxel (separate image for each hypothesis)

./output/ALE: outputs from coordinate-based meta-analysis, containing 1 - FDR-corrected p-value (separate image for each hypothesis)

./output/cluster_maps: average unthresholded statistical maps for clusters of teams (separate maps for each hypothesis/cluster)

./output/consensus_analysis: outputs from image-based meta-analysis, containing 1 - FDR-corrected p-value (separate image for each hypothesis)

./output/correlation_unthresh: correlation data for unthresholded maps
./output/correlation_unthresh/spearman_unthresh_hyp[1-9].csv: correlation matrix for unthresholded maps between teams for each hypothesis
./output/correlation_unthresh/unthresh_cluster_membership_spearman.json: stored cluster membership
./output/correlation_unthresh/mean_unthresh_correlation_by_cluster.csv: mean correlation between teams within each cluster

./output/unthresh_concat_zstat/hypo[1-9]_voxelmap.nii.gz: maps showing proportion of teams with data at each voxel

./cached/narps_prepare_maps.pkl: cached data structure for NARPS analysis

./logs/AnalyzeMaps-plot_distance_from_mean.txt:  results from analysis of distance from mean pattern 

./logs/AnalyzeMaps-mk_correlation_maps_unthresh.txt: results from analysis of number of nonzero voxels (unthresh zstat) and median correlation between teams 

./logs/narps.txt: main log file for analysis

./logs/ThresholdSimulation.log: results from simulation using consistent thresholding 

./logs/AnalyzeMaps-analyze_clusters.txt: results from cluster-specific analyses of unthresholded maps

./logs/AnalyzeMaps-get_thresh_similarity.txt: results from analysis of percent agreement and Jaccard similarity across thresholded maps

./logs/image_diagnostics.log: log from image processing 

./logs/ConsensusAnalysis.txt: results from image-based meta-analysis, including tau statistics 

./logs/AnalyzeMaps-mk_overlap_maps.txt: results from analysis of maximum voxel overlap 

./logs/zstat_diagnostics.log: log from processing of zstat maps 

./figures: analysis outputs and figures (note: for many of these figures there are both PDF and PNG versions present)

./figures/hyp[1-9]_combined_clusters.png: combined heatmap and cluster slices (separately for each hypothesis)

./figures/hyp[1-9]_pctagree_map_thresh.pdf: heatmaps for percent agreement

./figures/ModelingSummaryTable.tsv: summary of results from mixed effects logistic regression and nonparametric bootstrap analyses
./figures/OddsRatios.tsv

./figures/hyp[1-9]_ALE_z.png: Z statistic maps for coordinate-based meta-analysis
./figures/hyp[1-9]_ALE_p.png: uncorrected p-value maps for coordinate-based meta-analysis
./figures/hyp[1-9]_ALE_fdr_oneminusp.png: FDR corrected p-value maps for coordinate-based meta-analysis
./figures/hyp[1-9]_ALE_fdr_thresh_0.05.png: thresholded FDR corrected p-value maps for coordinate-based meta-analysis
./figures/hyp[1-9]_ALE_ale.png: Activation likelihood estimation maps for coordinate-based meta-analysis 
./figures/ALE_map.png

./figures/hyp[1-9]_spearman_map_unthresh.pdf: heatmap for spearman correlation of unthresholded maps between teams

./figures/range_map.png: maps of range in Z stats for all hypotheses

./figures/cluster_correlation.pdf: correlation map between all hypotheses/clusters

./figures/overlap_map.png: maps of overlap in thresholded maps for all hypotheses

./figures/hyp[1-9]_cluster_means.png: thresholded mean activation map for each cluster, separately for each hypothesis

./figures/tau_histograms.pdf: histograms of tau statistic across voxels for all hypotheses

./figures/tau_maps.pdf: tau maps for all hypotheses

./figures/ConfidenceDataWide.csv: confidence data for each team (used in generating final table)

./figures/median_corr_sorted.pdf: sorted median correlation for each team

./figures/std_map.png: maps of standard deviation across z statistics for all hypotheses

./figures/Table1.tsv: Summary of team reports (decision, confidence, similarity)

./figures/TomEtAl_correlation.pdf: correlation of each hypothesis/cluster with maps from original Tom et al. study

./figures/SuppFigure1.png: original version of figure showing all teams with modeling choices

./figures/ThresholdSimulation: results from simulations using consistent thresholding
./figures/ThresholdSimulation/simulation_results.csv: summary of overall results
./figures/ThresholdSimulation/ventralstriatum_mask.nii.gz: mask for ventral striatum
./figures/ThresholdSimulation/vmpfc_mask.nii.gz: mask for vmpfc
./figures/ThresholdSimulation/amygdala_mask.nii.gz: mask for amygdala
./figures/ThresholdSimulation/decision_vs_activation.png: plot of relative decision likelihood for original vs. re-thresholded data

./figures/MethodsTableMetadataMerged.csv: analysis metadata, using in creating final table

./figures/DecisionAnalysis.html: results from logistic regression analyses and bootstrap

./figures/NARPS_mean_correlation.pdf: correlation for each hypothesis/cluster with overall activation map from original NARPS analysis

./figures/DecisionDataWide.csv: decision data (used to generate final table)

./figures/consensus_map.pdf: maps from image-based meta-analysis

./figures/correlation_unthresh
./figures/correlation_unthresh/spearman_unthresh_hyp[1-9]_cluster[1-3].png: histograms of correlations for each hypothesis/cluster
./figures/correlation_unthresh/spearman_unthresh_hyp[1-9]_mean.png: histograms of correlations for each hypothesis overall

./PredictionMarkets: results from prediction market analyses
./PredictionMarkets/Figures/SuppTable_HoldingStats.tsv: Analyses of holdings
./PredictionMarkets/Figures/Active_vs_All_traders.pdf: Figure showing analysis of all traders vs. active traders
./PredictionMarkets/Figures/SuppTable_MarketDetails.tsv: Details regarding markets
./PredictionMarkets/Figures/SuppTable_MarketResults.tsv: Comparison of team vs. non-team market results
./PredictionMarkets/Figures/SuppTable_PanelRegressions.tsv: results from regression models on market results
./PredictionMarkets/Figures/PMbeliefs_Figure3.pdf: figure showing final market price vs. fundamental value
./PredictionMarkets/Figures/PM_Analyses.html: Results from Prediction market analyses
./PredictionMarkets/Figures/PM_Figures.html: Output from figure creation notebook
./PredictionMarkets/Figures/Timeseries.pdf: Price timeseries for each market

./PredictionMarkets/Combined: Data files used in Prediction Market analyses
./PredictionMarkets/Combined/position.RData
./PredictionMarkets/Combined/asset.RData
./PredictionMarkets/Combined/portfolio.RData
./PredictionMarkets/Combined/transaction.RData
./PredictionMarkets/Combined/teamresults.RData

./PredictionMarkets/Processed: Intermediate data files used in Prediction Market analyses
./PredictionMarkets/Processed/Fundamentals.RData
./PredictionMarkets/Processed/Prices.RData
./PredictionMarkets/Processed/BalancedPanel.RData
./PredictionMarkets/Processed/Transactions.RData
./PredictionMarkets/Processed/TeamResults.RData
./PredictionMarkets/Processed/Holdings.RData

./metadata: various metadata outputs
./metadata/cluster_corr_TomEtAl.csv: correlation between each hypothesis/cluster and Tom et al. original maps
./metadata/cluster_metadata_df.csv: cluster membership by team/hypothesis
./metadata/thresh_voxel_statistics.csv: statistics on thresholded maps for each hypothesis
./metadata/smoothness_est.csv: smoothness estimates per hypothesis/team
./metadata/all_metadata.csv: full metadata output
./metadata/rectified_images_list.txt: list of teams/hypotheses that required direction-flipping
./metadata/cluster_corr_NARPS_mean.csv: correlation of each hypothesis/cluster with NARPS overall task activation maps
./metadata/pctagree_hyp[1-9].csv: percent agreement data across teams
./metadata/image_metadata_df.csv: number of NA and nonzero voxels for each team
./metadata/narps_metadata_all_teams.csv: original metadata for all teams
./metadata/median_pattern_corr.csv: median correlation for each team with mean map across teams
./metadata/thresh_voxel_data.csv: metadata for each team's thresholded maps
./metadata/R_package_info.txt: listing of all R package/versions used in analysis
