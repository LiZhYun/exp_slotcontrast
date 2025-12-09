#!/bin/bash

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

    "baseline_w_contrast false false false false true 0.5 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.CrossAttentionPredictor"
    "baseline_wo_contrast false false false false true 0.0 0.0 0 vit_block12 both GreedyFeatureInit per_frame networks.CrossAttentionPredictor"
    

)

for config in "${CONFIGS[@]}"; do
    read -r exp_name use_ttt3r use_gated use_gated_predictor use_ttt use_gru loss_ss loss_cycle window_size features cross_mode init_name init_mode predictor <<< "$config"

    echo "Submitting: $exp_name"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh \
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
        "model.predictor.name=${predictor}"
done

echo "All jobs submitted!"
