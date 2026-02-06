#!/bin/bash

# ytvis2021.yaml, movi_c.yaml, movi_d.yaml, movi_e.yaml
CONFIG_FILE="/project/project_462001066/exp_slotcontrast/configs/slotcontrast/movi_e.yaml"


CONFIGS=(

    # # ytvis2021
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 42 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 43 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 44 2 3 local_consistency false 0.5"

    # # TODO
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 42 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 43 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 44 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 42 2 3 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 43 2 3 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 44 2 3 local_consistency false 1.5"

    # # one iteration
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 42 1 1 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 43 1 1 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 44 1 1 local_consistency false 0.5"

    # # baseline
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 2 42 2 3 local_consistency false 0.5"
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 2 43 2 3 local_consistency false 0.5"
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 2 44 2 3 local_consistency false 0.5"

    # # rethinking slot init strategies
    # # slot init first frame with greedy
    # "first_frame_greedy_w_contrast GreedyFeatureInit first_frame networks.TransformerEncoder 2 42 2 3 local_consistency false 0.5"
    # "first_frame_greedy_w_contrast GreedyFeatureInit first_frame networks.TransformerEncoder 2 43 2 3 local_consistency false 0.5"
    # "first_frame_greedy_w_contrast GreedyFeatureInit first_frame networks.TransformerEncoder 2 44 2 3 local_consistency false 0.5"
    # "first_frame_greedy_w_contrast GreedyFeatureInit first_frame networks.TransformerEncoder 2 42 1 1 local_consistency false 0.5"
    # "first_frame_greedy_w_contrast GreedyFeatureInit first_frame networks.TransformerEncoder 2 43 1 1 local_consistency false 0.5"
    # "first_frame_greedy_w_contrast GreedyFeatureInit first_frame networks.TransformerEncoder 2 44 1 1 local_consistency false 0.5"

    # # greedy init perframe but skip predictor
    # "per_frame_greedy_wo_pre_w_contrast GreedyFeatureInit per_frame networks.TransformerEncoder 2 42 2 3 local_consistency true 0.5"
    # "per_frame_greedy_wo_pre_w_contrast GreedyFeatureInit per_frame networks.TransformerEncoder 2 43 2 3 local_consistency true 0.5"
    # "per_frame_greedy_wo_pre_w_contrast GreedyFeatureInit per_frame networks.TransformerEncoder 2 44 2 3 local_consistency true 0.5"

    # # rethinking predictor
    # # fix init skip predictor
    # "baseline_wo_pre_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 2 42 2 3 local_consistency true 0.5"
    # "baseline_wo_pre_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 2 43 2 3 local_consistency true 0.5"
    # "baseline_wo_pre_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 2 44 2 3 local_consistency true 0.5"

    # # fix init replace predictor with hungarian
    # "baseline_hungarian_w_contrast FixedLearnedInit first_frame networks.HungarianPredictor 2 42 2 3 local_consistency false 0.5"
    # "baseline_hungarian_w_contrast FixedLearnedInit first_frame networks.HungarianPredictor 2 43 2 3 local_consistency false 0.5"
    # "baseline_hungarian_w_contrast FixedLearnedInit first_frame networks.HungarianPredictor 2 44 2 3 local_consistency false 0.5"

    # # ablation
    # # saliency mode
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 42 2 3 norm false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 43 2 3 norm false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 44 2 3 norm false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 42 2 3 pca false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 43 2 3 pca false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 44 2 3 pca false 0.5"
    # # neighbor_radius
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 3 42 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 3 43 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 3 44 2 3 local_consistency false 0.5"


    # # movi_c
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 1.5"

    # # one iteration
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 1 1 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 1 1 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 1 1 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 1 1 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 1 1 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 1 1 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 1 1 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 1 1 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 1 1 local_consistency false 1.5"

    # # baseline
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 1 42 2 3 local_consistency false 1.0"
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 1 43 2 3 local_consistency false 1.0"
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 1 44 2 3 local_consistency false 1.0"


    # # movi_d
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 1.5"

    # # one iteration
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 1 1 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 1 1 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 1 1 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 1 1 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 1 1 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 1 1 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 1 1 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 1 1 local_consistency false 1.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 1 1 local_consistency false 1.5"

    # # baseline
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 1 42 2 3 local_consistency false 1.0"
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 1 43 2 3 local_consistency false 1.0"
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 1 44 2 3 local_consistency false 1.0"


    # # movi_e
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 1.0"

    # # TODO
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 0.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 0.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 0.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 1.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 1.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 1.5"

    # # one iteration
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 1 1 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 1 1 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 1 1 local_consistency false 1.0"

    # # TODO
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 1 1 local_consistency false 0.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 1 1 local_consistency false 0.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 1 1 local_consistency false 0.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 1 1 local_consistency false 1.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 1 1 local_consistency false 1.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 1 1 local_consistency false 1.5"

    # # baseline
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 1 42 2 3 local_consistency false 1.0"
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 1 43 2 3 local_consistency false 1.0"
    # "baseline_w_contrast FixedLearnedInit first_frame networks.TransformerEncoder 1 44 2 3 local_consistency false 1.0"

    # # ablation
    # # saliency mode
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 norm false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 norm false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 norm false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 pca false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 pca false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 pca false 1.0"
    # # neighbor_radius
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 42 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 43 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 44 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 3 42 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 3 43 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 3 44 2 3 local_consistency false 1.0"


    )
    
for config in "${CONFIGS[@]}"; do
    read -r exp_name init_name init_mode predictor neighbor_radius SEED n_iters f_n_iters saliency_mode skip_predictor saliency_alpha<<< "$config"

    echo "Submitting: $exp_name with config: $CONFIG_FILE"
    
    sbatch --job-name="sc_${exp_name}" lumi_slurm.sh "${CONFIG_FILE}" \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.initializer.name=${init_name}" \
        "model.initializer.init_mode=${init_mode}" \
        "model.initializer.neighbor_radius=${neighbor_radius}" \
        "model.initializer.saliency_mode=${saliency_mode}" \
        "model.initializer.saliency_alpha=${saliency_alpha}" \
        "model.predictor.name=${predictor}" \
        "model.grouper.n_iters=${n_iters}" \
        "model.latent_processor.first_step_corrector_args.n_iters=${f_n_iters}" \
        "model.latent_processor.skip_predictor=${skip_predictor}" \
        "seed=${SEED}"
done

echo "All jobs submitted!"
