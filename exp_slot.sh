#!/bin/bash

# Dataset configuration - change this to switch datasets
# Available configs: configs/slotcontrast/ytvis2021.yaml, configs/slotcontrast/movi_c.yaml, configs/slotcontrast/movi_e.yaml
CONFIG_FILE="configs/slotcontrast/movi_e.yaml"

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

    "per_frame_greedy_hungarian_w_contrast 0.5 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12"
    "per_frame_greedy_hungarian_wo_contrast 0.0 GreedyFeatureInit per_frame networks.HungarianPredictor false false 2 3 false vit_block12"

    )

SEED=42
    
for config in "${CONFIGS[@]}"; do
    read -r exp_name loss_ss init_name init_mode predictor skip_corrector skip_predictor n_iters f_n_iters use_gated features <<< "$config"

    echo "Submitting: $exp_name with config: $CONFIG_FILE"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh "${CONFIG_FILE}" \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.loss_weights.loss_ss=${loss_ss}" \
        "model.initializer.name=${init_name}" \
        "model.initializer.init_mode=${init_mode}" \
        "model.predictor.name=${predictor}" \
        "model.latent_processor.skip_corrector=${skip_corrector}" \
        "model.latent_processor.skip_predictor=${skip_predictor}" \
        "model.grouper.n_iters=${n_iters}" \
        "model.latent_processor.first_step_corrector_args.n_iters=${f_n_iters}" \
        "model.grouper.use_gated=${use_gated}" \
        "model.encoder.backbone.features=${features}" \
        "seed=${SEED}"
done

echo "All jobs submitted!"
