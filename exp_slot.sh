#!/bin/bash

# Define experiment configurations
# Format: "experiment_name use_ttt3r use_gated use_gated_predictor use_ttt use_gru loss_ss loss_cycle"
CONFIGS=(
    # "baseline_w_contrast false false false false true 0.5 0.0"
    # "baseline_wo_contrast false false false false true 0.0 0.0"

    # "gated_attention_grouper_w_contrast false true false false false 0.5 0.0" # may work
    # "gated_attention_grouper_wo_contrast false true false false false 0.0 0.0"

#     "gated_attention_grouper_gru_w_contrast false true false false true 0.5 0.0"

    # "gated_attention_predictor_w_contrast false false true false true 0.5 0.0"
    # "gated_attention_predictor_wo_contrast false false true false true 0.0 0.0" 

    # "gated_attention_grouper_predictor_w_contrast false true true false false 0.5 0.0" # may work
    # "gated_attention_grouper_predictor_wo_contrast false true true false false 0.0 0.0"

#     "gated_attention_grouper_predictor_gru_w_contrast false true true false true 0.5 0.0"

    # "ttt3r_grouper_w_contrast false false false true false 0.5 0.0"
    # "ttt3r_grouper_wo_contrast false false false true false 0.0 0.0"

    # "ttt3r_predictor_w_contrast true false false false true 0.5 0.0"
    # "ttt3r_predictor_wo_contrast true false false false true 0.0 0.0"

    # "ttt3r_grouper_predictor_w_contrast true false false true false 0.5 0.0" # may work
    # "ttt3r_grouper_predictor_wo_contrast true false false true false 0.0 0.0"

#     "ttt3r_grouper_predictor_gru_w_contrast true false false true true 0.5 0.0"

    "loss_cycle_w_contrast false false false false true 0.5 0.5"
    "loss_cycle_wo_contrast false false false false true 0.0 0.5"
)

for config in "${CONFIGS[@]}"; do
    read -r exp_name use_ttt3r use_gated use_gated_predictor use_ttt use_gru loss_ss loss_cycle <<< "$config"

    echo "Submitting: $exp_name"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.latent_processor.use_ttt3r=${use_ttt3r}" \
        "model.grouper.use_gated=${use_gated}" \
        "model.predictor.use_gated=${use_gated_predictor}" \
        "model.grouper.use_ttt=${use_ttt}" \
        "model.grouper.use_gru=${use_gru}" \
        "model.loss_weights.loss_ss=${loss_ss}" \
        "model.loss_weights.loss_cycle=${loss_cycle}"
done

echo "All jobs submitted!"
