#!/bin/bash

# Dataset configuration - change this to switch datasets
# Available configs: configs/slotcontrast/ytvis2021.yaml, configs/slotcontrast/movi_c.yaml, configs/slotcontrast/movi_e.yaml
CONFIG_FILE="configs/slotcontrast/ytvis2021.yaml"

# Define experiment configurations
# Format: "experiment_name use_ttt3r use_gated use_gated_predictor use_ttt use_gru loss_ss loss_cycle window_size"
CONFIGS=(
    # "baseline_w_contrast false false false false true 0.5 0.0 0"
    # "baseline_wo_contrast false false false false true 0.0 0.0 0"

    # "gated_attention_grouper_w_contrast false true false false false 0.5 0.0 0"
    # "gated_attention_grouper_wo_contrast false true false false false 0.0 0.0 0"

    # "gated_attention_grouper_gru_w_contrast false true false false true 0.5 0.0 0"

    # "gated_attention_predictor_w_contrast false false true false true 0.5 0.0 0"
    # "gated_attention_predictor_wo_contrast false false true false true 0.0 0.0 0"

    # "gated_attention_grouper_predictor_w_contrast false true true false false 0.5 0.0 0"
    # "gated_attention_grouper_predictor_wo_contrast false true true false false 0.0 0.0 0"

    # "gated_attention_grouper_predictor_gru_w_contrast false true true false true 0.5 0.0 0"

    # "ttt3r_grouper_w_contrast false false false true false 0.5 0.0 0"
    # "ttt3r_grouper_wo_contrast false false false true false 0.0 0.0 0"

    # "ttt3r_predictor_w_contrast true false false false true 0.5 0.0 0"
    # "ttt3r_predictor_wo_contrast true false false false true 0.0 0.0 0"

    # "ttt3r_grouper_predictor_w_contrast true false false true false 0.5 0.0 0"
    # "ttt3r_grouper_predictor_wo_contrast true false false true false 0.0 0.0 0"

    # "ttt3r_grouper_predictor_gru_w_contrast true false false true true 0.5 0.0 0"

    # "loss_cycle_w_contrast false false false false true 0.5 0.5 0"
    # "loss_cycle_wo_contrast false false false false true 0.0 0.5 0"

    # "loss_cycle_w_contrast false false false false true 0.5 1.0 0"
    # "loss_cycle_w_contrast false false false false true 0.5 0.8 0"
    # "loss_cycle_w_contrast false false false false true 0.5 0.2 0"
    # "loss_cycle_w_contrast false false false false true 0.5 0.1 0"

    # "loss_cycle_wo_contrast false false false false true 0.0 1.0 0"
    # "loss_cycle_wo_contrast false false false false true 0.0 0.8 0"
    # "loss_cycle_wo_contrast false false false false true 0.0 0.2 0"
    # "loss_cycle_wo_contrast false false false false true 0.0 0.1 0"

    # "window_loss_cycle_w_contrast false false false false true 0.5 0.5 1"
    # "window_loss_cycle_w_contrast false false false false true 0.5 0.5 2"
    # "window_loss_cycle_w_contrast false false false false true 0.5 0.5 3"

    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 1"
    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 2"
    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 3"

    # "gated_attention_grouper_w_contrast false true false false false 0.5 0.5 0"
    # "gated_attention_grouper_w_contrast false true false false false 0.1 0.5 0"
    # "gated_attention_grouper_wo_contrast false true false false false 0.0 0.5 0"

    # "ttt3r_grouper_predictor_gru_w_contrast true false false true true 0.5 0.5 0"
    # "ttt3r_grouper_predictor_gru_wo_contrast true false false true true 0.0 0.5 0"

    # "window_loss_cycle_w_contrast false false false false true 0.1 0.5 2"

    # "gated_attention_grouper_gru_wo_contrast_cycle false true false false true 0.0 0.5 0"

    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block6"
    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block7"
    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block8"
    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block9"
    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block10"

    # "window_loss_cycle_w_contrast false false false false true 0.5 0.5 2 vit_block11 backward"
    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 2 vit_block11 backward"

    # "window_loss_cycle_w_contrast false false false false true 0.5 0.5 2 vit_block11 forward"
    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 2 vit_block11 forward"

    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block6 both GreedyFeatureInit first_frame"
    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block6 both GreedyFeatureInit per_frame"

    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block6 both GreedyFeatureInit first_frame"
    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block6 both GreedyFeatureInit per_frame"

    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit first_frame"
    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame"

    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit first_frame"
    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame"

    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.CrossAttentionPredictor"
    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.CrossAttentionPredictor"
    
    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block11 both FixedLearnedInit first_frame networks.TransformerEncoder"

    # "window_loss_cycle_w_contrast false false false false true 0.5 0.5 2 vit_block12 forward FixedLearnedInit first_frame networks.TransformerEncoder"
    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 2 vit_block12 backward FixedLearnedInit first_frame networks.TransformerEncoder"
    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 2 vit_block12 forward FixedLearnedInit first_frame networks.TransformerEncoder"

    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    
    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit first_frame networks.TransformerEncoder"

    # "gated_attention_grouper_wo_contrast false true false false false 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "gated_attention_predictor_w_contrast false false true false true 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "gated_attention_predictor_wo_contrast false false true false true 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "gated_attention_grouper_predictor_wo_contrast false true true false false 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"

    # "ttt3r_grouper_w_contrast false false false true false 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "ttt3r_grouper_wo_contrast false false false true false 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "ttt3r_predictor_w_contrast true false false false true 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "ttt3r_predictor_wo_contrast true false false false true 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "ttt3r_grouper_predictor_wo_contrast true false false true false 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"

    # "gated_attention_grouper_gru_w_contrast false true false false true 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "gated_attention_grouper_gru_wo_contrast false true false false true 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"

    # "gated_attention_grouper_w_contrast false true false false false 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"

    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    
    # "gated_attention_grouper_predictor_gru_w_contrast false true true false true 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "gated_attention_grouper_predictor_gru_wo_contrast false true true false true 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    
    # "ttt3r_grouper_predictor_gru_w_contrast true false false true true 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "ttt3r_grouper_predictor_gru_wo_contrast true false false true true 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"

    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block11 both FixedLearnedInit first_frame networks.TransformerEncoder"

    # "gated_attention_grouper_wo_contrast false true false false false 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"

    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "window_loss_cycle_w_contrast false false false false true 0.5 1.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "window_loss_cycle_wo_contrast false false false false true 0.0 1.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "window_loss_cycle_w_contrast false false false false true 0.1 1.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "window_loss_cycle_w_contrast false false false false true 0.1 0.5 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"
    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 2 vit_block11 forward FixedLearnedInit first_frame networks.TransformerEncoder"

    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor"
    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor"

    # "window_loss_cycle_wo_contrast false false false false true 0.0 0.5 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder"

    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor true"
    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor true"
    
    # "3d_pos_emb_w_contrast false false false false true 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder false true"
    # "3d_pos_emb_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder false true"

    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor true false true"
    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor true false true"
    
    # "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"
    # "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"

    # "hungarian_pre_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"
    # "hungarian_pre_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"

    # "hungarian_pre_block11_w_contrast false false false false true 0.5 0.0 0 vit_block11 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"
    # "hungarian_pre_block11_wo_contrast false false false false true 0.0 0.0 0 vit_block11 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"

    # "hungarian_pre_block12_pca_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"
    # "hungarian_pre_block12_pca_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"

    # "hungarian_greedy_block12_norm_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"
    # "hungarian_greedy_block12_norm_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.HungarianPredictor false false false"

    # "no_predictor_w_contrast false false false false true 0.5 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder false false false"
    # "no_predictor_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both FixedLearnedInit first_frame networks.TransformerEncoder false false false"

    "per_frame_no_predictor_w_contrast false false false false true 0.5 0.0 0 vit_block12 both FixedLearnedInit per_frame networks.TransformerEncoder false false false true"
    "per_frame_no_predictor_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both FixedLearnedInit per_frame networks.TransformerEncoder false false false true"

    )

SEED=42
    
for config in "${CONFIGS[@]}"; do
    read -r exp_name use_ttt3r use_gated use_gated_predictor use_ttt use_gru loss_ss loss_cycle window_size features cross_mode init_name init_mode predictor skip_corrector use_pos_embed aggregate skip_predictor <<< "$config"

    echo "Submitting: $exp_name with config: $CONFIG_FILE"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh "${CONFIG_FILE}" \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.latent_processor.use_ttt3r=${use_ttt3r}" \
        "model.grouper.use_gated=${use_gated}" \
        "model.predictor.use_gated=${use_gated_predictor}" \
        "model.grouper.use_ttt=${use_ttt}" \
        "model.grouper.use_gru=${use_gru}" \
        "model.loss_weights.loss_ss=${loss_ss}" \
        "model.loss_weights.loss_cycle=${loss_cycle}" \
        "model.temporal_cross_window=${window_size}" \
        "model.encoder.backbone.features=${features}" \
        "model.temporal_cross_mode=${cross_mode}" \
        "model.initializer.name=${init_name}" \
        "model.initializer.init_mode=${init_mode}" \
        "model.predictor.name=${predictor}" \
        "model.latent_processor.skip_corrector=${skip_corrector}" \
        "model.latent_processor.skip_predictor=${skip_predictor}" \
        "model.encoder.use_pos_embed=${use_pos_embed}" \
        "model.initializer.aggregate=${aggregate}" \
        "seed=${SEED}"
done

echo "All jobs submitted!"
