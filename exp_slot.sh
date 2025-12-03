#!/bin/bash

# Define experiment configurations
# Format: "experiment_name use_ttt3r use_gated use_ttt use_gru loss_ss"
CONFIGS=(
    "baseline_w_contrast false false false true 0.5"
    "baseline_wo_contrast false false false true 0.0"

    "gated_attention_grouper_w_contrast false true false false 0.5"
    "gated_attention_grouper_wo_contrast false true false false 0.0"

    "ttt3r_grouper_w_contrast false false true false 0.5"
    "ttt3r_grouper_wo_contrast false false true false 0.0"

    "ttt3r_predictor_w_contrast true false false true 0.5"
    "ttt3r_predictor_wo_contrast true false false true 0.0"
)

for config in "${CONFIGS[@]}"; do
    read -r exp_name use_ttt3r use_gated use_ttt use_gru loss_ss <<< "$config"
    
    echo "Submitting: $exp_name"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.latent_processor.use_ttt3r=${use_ttt3r}" \
        "model.grouper.use_gated=${use_gated}" \
        "model.grouper.use_ttt=${use_ttt}" \
        "model.grouper.use_gru=${use_gru}" \
        "model.loss_weights.loss_ss=${loss_ss}"
done

echo "All jobs submitted!"
