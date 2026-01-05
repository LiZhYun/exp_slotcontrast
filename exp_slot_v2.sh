#!/bin/bash

# Dataset configuration - change this to switch datasets
# Available configs: configs/slotcontrast/ytvis2021.yaml, configs/slotcontrast/movi_c.yaml, configs/slotcontrast/movi_e.yaml
CONFIG_FILE="configs/slotcontrast/ytvis2021.yaml"

# Define experiment configurations
# Format: "experiment_name use_ttt3r use_gated use_gated_predictor use_ttt use_gru loss_ss loss_cycle window_size"
CONFIGS=(

    # "baseline_w_contrast 0.5 FixedLearnedInit first_frame networks.TransformerEncoder false false 2 3"
    # "baseline_wo_contrast 0.0 FixedLearnedInit first_frame networks.TransformerEncoder false false 2 3"

    # "no_predictor_w_contrast 0.5 FixedLearnedInit first_frame networks.TransformerEncoder false true 2 3"
    # "no_predictor_wo_contrast 0.0 FixedLearnedInit first_frame networks.TransformerEncoder false true 2 3"

    # "no_predictor_w_contrast 0.5 FixedLearnedInit first_frame networks.TransformerEncoder false true 1 1"
    # "no_predictor_wo_contrast 0.0 FixedLearnedInit first_frame networks.TransformerEncoder false true 1 1"

    # "per_frame_no_predictor_w_contrast 0.5 FixedLearnedInit per_frame networks.TransformerEncoder false true 2 3"
    # "per_frame_no_predictor_wo_contrast 0.0 FixedLearnedInit per_frame networks.TransformerEncoder false true 2 3"

    # "greedy_no_predictor_w_contrast 0.5 GreedyFeatureInit per_frame networks.TransformerEncoder false true 2 3"
    # "greedy_no_predictor_wo_contrast 0.0 GreedyFeatureInit per_frame networks.TransformerEncoder false true 2 3"

    # "greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3"
    # "greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3"

    # "greedy_no_predictor_w_contrast 0.5 GreedyFeatureInit per_frame networks.TransformerEncoder false true 1 1"
    # "greedy_no_predictor_wo_contrast 0.0 GreedyFeatureInit per_frame networks.TransformerEncoder false true 1 1"

    # "per_frame_no_predictor_w_contrast 0.5 FixedLearnedInit per_frame networks.TransformerEncoder false true 1 1"
    # "per_frame_no_predictor_wo_contrast 0.0 FixedLearnedInit per_frame networks.TransformerEncoder false true 1 1"

    # "baseline_w_contrast 0.5 FixedLearnedInit first_frame networks.TransformerEncoder false false 1 1"
    # "baseline_wo_contrast 0.0 FixedLearnedInit first_frame networks.TransformerEncoder false false 1 1"

    # "first_frame_greedy_w_contrast 0.5 GreedyFeatureInit first_frame networks.TransformerEncoder false false 2 3"
    # "first_frame_greedy_wo_contrast 0.0 GreedyFeatureInit first_frame networks.TransformerEncoder false false 2 3"

    # "first_frame_greedy_w_contrast 0.5 GreedyFeatureInit first_frame networks.TransformerEncoder false false 1 1"
    # "first_frame_greedy_wo_contrast 0.0 GreedyFeatureInit first_frame networks.TransformerEncoder false false 1 1"

    # "first_frame_greedy_no_predictor_w_contrast 0.5 GreedyFeatureInit first_frame networks.TransformerEncoder false true 2 3"
    # "first_frame_greedy_no_predictor_wo_contrast 0.0 GreedyFeatureInit first_frame networks.TransformerEncoder false true 2 3"

    # "first_frame_greedy_no_predictor_w_contrast 0.5 GreedyFeatureInit first_frame networks.TransformerEncoder false true 1 1"
    # "first_frame_greedy_no_predictor_wo_contrast 0.0 GreedyFeatureInit first_frame networks.TransformerEncoder false true 1 1"

    # "first_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit first_frame networks.HungarianPredictor false false 2 3"
    # "first_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit first_frame networks.HungarianPredictor false false 2 3"

    # "first_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit first_frame networks.HungarianPredictor false false 1 1"
    # "first_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit first_frame networks.HungarianPredictor false false 1 1"

    # "first_frame_greedy_w_contrast 0.5 GreedyFeatureInit first_frame networks.TransformerEncoder false false 2 3 true vit_block11"
    # "first_frame_greedy_wo_contrast 0.0 GreedyFeatureInit first_frame networks.TransformerEncoder false false 2 3 true vit_block11"

    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12"

    # "first_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit first_frame networks.HungarianPredictor false false 2 3 false vit_block12"
    # "first_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit first_frame networks.HungarianPredictor false false 2 3 false vit_block12"

    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12"

    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.5"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.5"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca true 0.5"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca true 0.5"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca true 0.0"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca true 0.0"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 norm true 0.5"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 norm true 0.5"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 norm true 0.0"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 norm true 0.0"

    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca false 0.5"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca false 0.5"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca true 0.5"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca true 0.5"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca true 0.8"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca true 0.8"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca true 0.25"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca true 0.25"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca true 0.0"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 pca true 0.0"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca true 0.8"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca true 0.8"
    # "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca true 0.25"
    # "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca true 0.25"

    # "per_frame_cluster_hungarian_w_contrast 0.5 ClusterFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.25"
    # "per_frame_cluster_hungarian_wo_contrast 0.0 ClusterFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.25"
    # "per_frame_cluster_hungarian_w_contrast 0.5 ClusterFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.25 kmeans"
    # "per_frame_cluster_hungarian_wo_contrast 0.0 ClusterFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.25 kmeans"
    # "per_frame_cluster_hungarian_w_contrast 0.5 ClusterFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.25 spectral"
    # "per_frame_cluster_hungarian_wo_contrast 0.0 ClusterFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.25 spectral"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_pca false 0.5 kmeans 1"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_pca false 0.5 kmeans 1"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_pca false 0.5 kmeans 2"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_pca false 0.5 kmeans 2"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 3 0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 3 0"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency true 0.0 kmeans 1 0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency true 0.0 kmeans 1 0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency true 0.25 kmeans 1 0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency true 0.25 kmeans 1 0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency true 0.5 kmeans 1 0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency true 0.5 kmeans 1 0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency true 0.8 kmeans 1 0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency true 0.8 kmeans 1 0"

    # "per_frame_greedy_local_soomth_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 1"
    # "per_frame_greedy_local_soomth_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 1"
    # "per_frame_greedy_local_soomth_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 2"
    # "per_frame_greedy_local_soomth_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 2"
    # "per_frame_greedy_local_soomth_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 3"
    # "per_frame_greedy_local_soomth_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 3"

    # "per_frame_greedy_local_soft_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 soft"
    # "per_frame_greedy_local_soft_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 soft"
    # "per_frame_greedy_local_soft_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 neighbor_avg"
    # "per_frame_greedy_local_soft_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 neighbor_avg"

    # "per_frame_greedy_local_ms_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_ms false 0.5 kmeans 1 0 hard"
    # "per_frame_greedy_local_ms_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_ms false 0.5 kmeans 1 0 hard"
    # "per_frame_greedy_local_uniform_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_uniform false 0.5 kmeans 1 0 hard"
    # "per_frame_greedy_local_uniform_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_uniform false 0.5 kmeans 1 0 hard"
    # "per_frame_greedy_local_centroid_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_centroid false 0.5 kmeans 1 0 hard"
    # "per_frame_greedy_local_centroid_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_centroid false 0.5 kmeans 1 0 hard"
    # "per_frame_greedy_local_balanced_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_balanced false 0.5 kmeans 1 0 hard"
    # "per_frame_greedy_local_balanced_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_balanced false 0.5 kmeans 1 0 hard"
    # "per_frame_greedy_local_soft_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_soft false 0.5 kmeans 1 0 hard"
    # "per_frame_greedy_local_soft_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_soft false 0.5 kmeans 1 0 hard"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 0.5"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 0.7"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 0.7"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 2 0 hard 1.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 2 0 hard 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.3"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.3"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.5"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.5"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 2.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 2.0"

    # "per_frame_greedy_local_spatial_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 1 0.5"
    # "per_frame_greedy_local_spatial_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 1 0.5"
    # "per_frame_greedy_local_spatial_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 1 1.0"
    # "per_frame_greedy_local_spatial_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 1 1.0"
    # "per_frame_greedy_local_spatial_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 2 0.5"
    # "per_frame_greedy_local_spatial_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 2 0.5"
    # "per_frame_greedy_local_spatial_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 2 1.0"
    # "per_frame_greedy_local_spatial_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 2 1.0"
    # "per_frame_greedy_local_spatial_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 1 0.5"
    # "per_frame_greedy_local_spatial_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 1 0.5"
    # "per_frame_greedy_local_spatial_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 1 1.0"
    # "per_frame_greedy_local_spatial_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 1 1.0"
    # "per_frame_greedy_local_spatial_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 2 0.5"
    # "per_frame_greedy_local_spatial_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 2 0.5"
    # "per_frame_greedy_local_spatial_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 2 1.0"
    # "per_frame_greedy_local_spatial_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 2 1.0"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_density false 0.5 kmeans 1 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_density false 0.5 kmeans 1 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_second false 0.5 kmeans 1 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_second false 0.5 kmeans 1 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_gaussian false 0.5 kmeans 1 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_gaussian false 0.5 kmeans 1 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 knn_refine 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 knn_refine 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 centroid 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 centroid 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_density false 0.5 kmeans 2 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_density false 0.5 kmeans 2 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_second false 0.5 kmeans 2 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_second false 0.5 kmeans 2 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_gaussian false 0.5 kmeans 2 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency_gaussian false 0.5 kmeans 2 0 hard 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 knn_refine 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 knn_refine 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 centroid 1.0 0 0.0"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 centroid 1.0 0 0.0"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 true"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 true"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 true"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 true"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 true"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 true"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 true"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 true"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 true 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 true 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 true 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 true 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 true 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 true 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 true 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block11 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 true 64"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 4 4 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 4 4 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 3 4 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 3 4 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 3 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 3 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 2 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 2 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 1 2 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 1 2 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 1 1 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 1 1 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 4 4 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 4 4 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 3 4 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 3 4 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 3 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 3 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 2 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 2 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 1 2 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 1 2 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 1 1 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 1 1 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64"

    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 true"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 true"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 true"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 true"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 true"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 true"
    # "per_frame_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 true"
    # "per_frame_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12 pca false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 true"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.1 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.1 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.1 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.1 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.2 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.2 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.2 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.2 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.3 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.3 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.3 0.3 20"
    # "per_frame_vari_greedy_local_mem_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.3 0.3 20"

    # # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 15"
    # # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 15"
    # # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 20"
    # # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 20"

    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 20"

    # # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 7"
    # # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 7"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 7"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 7"
    # # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 20"
    # # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianPredictor false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 20"

    # # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 7"
    # # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 7"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 7"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 7"
    # # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 20"
    # # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0005 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 20"
    # "per_frame_vari_greedy_local_hungarian_wo_contrast 0.0 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0001 0.3 20"
    
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0 2.0 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 2.0 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.1 2.0 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.3 2.0 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0 0.3 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0 0.6 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.0 1.0 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 1.0 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 1.5 15"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 1.0 20"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 1 0 hard 1.0 0 0.0 false 64 false 0.01 1.5 20"

    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0 2.0 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 2.0 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.1 2.0 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.3 2.0 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0 0.3 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0 0.6 7"
    # "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.0 1.0 7"
    "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 1.0 7"
    "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 1.5 7"
    "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 1.0 20"
    "per_frame_vari_greedy_local_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher false false 2 3 false vit_block12 local_consistency false 0.5 kmeans 2 0 hard 1.0 0 0.0 false 64 false 0.01 1.5 20"


    )

SEED=42
    
for config in "${CONFIGS[@]}"; do
    read -r exp_name loss_ss init_name init_mode predictor skip_corrector skip_predictor n_iters f_n_iters use_gated features saliency_mode aggregate aggregate_threshold cluster_method neighbor_radius saliency_smoothing selection_mode saliency_alpha spatial_suppression_radius spatial_suppression_strength refine_linear refine_hidden use_backbone_features init_threshold match_threshold NUM_SLOTS <<< "$config"

    echo "Submitting: $exp_name with config: $CONFIG_FILE"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh "${CONFIG_FILE}" \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.loss_weights.loss_ss=${loss_ss}" \
        "model.initializer.name=${init_name}" \
        "model.initializer.init_mode=${init_mode}" \
        "model.initializer.saliency_mode=${saliency_mode}" \
        "model.initializer.aggregate=${aggregate}" \
        "model.initializer.aggregate_threshold=${aggregate_threshold}" \
        "model.initializer.cluster_method=${cluster_method}" \
        "model.initializer.neighbor_radius=${neighbor_radius}" \
        "model.initializer.saliency_smoothing=${saliency_smoothing}" \
        "model.initializer.selection_mode=${selection_mode}" \
        "model.initializer.saliency_alpha=${saliency_alpha}" \
        "model.initializer.spatial_suppression_radius=${spatial_suppression_radius}" \
        "model.initializer.spatial_suppression_strength=${spatial_suppression_strength}" \
        "model.initializer.refine_linear=${refine_linear}" \
        "model.initializer.refine_hidden=${refine_hidden}" \
        "model.initializer.init_threshold=${init_threshold}" \
        "model.predictor.name=${predictor}" \
        "model.predictor.match_threshold=${match_threshold}" \
        "model.latent_processor.skip_corrector=${skip_corrector}" \
        "model.latent_processor.skip_predictor=${skip_predictor}" \
        "model.grouper.n_iters=${n_iters}" \
        "model.latent_processor.first_step_corrector_args.n_iters=${f_n_iters}" \
        "model.grouper.use_gated=${use_gated}" \
        "model.encoder.backbone.features=${features}" \
        "model.use_backbone_features=${use_backbone_features}" \
        "globals.NUM_SLOTS=${NUM_SLOTS}" \
        "seed=${SEED}"
done

echo "All jobs submitted!"
