#!/bin/bash

# ytvis2021.yaml, movi_c.yaml, movi_d.yaml, movi_e.yaml
CONFIG_FILE="configs/slotcontrast/movi_e.yaml"


CONFIGS=(
    # # ytvis2021
    # "42 gcv2_dinov2_block12 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518"
    # "42 gcv2_dinov2_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch14_dinov2 1369 518"
    # "42 gcv2_learnable_pos_emb_dinov2_block12 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518"
    # "42 gcv2_learnable_pos_emb_dinov2_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output  vit_output vit_base_patch14_dinov2 1369 518"

    # "42 gcv2_dinov3_block12 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512"
    # "42 gcv2_dinov3_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch16_dinov3 1024 512"
    # "42 gcv2_learnable_pos_emb_dinov3_block12 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512"
    # "42 gcv2_learnable_pos_emb_dinov3_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_base_patch16_dinov3 1024 512"
    
    # # # baseline
    # "42 baseline_dinov2_block12 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518"
    # "42 baseline_dinov2_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch14_dinov2 1369 518"
    # "42 baseline_learnable_pos_emb_dinov2_block12 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518"
    # "42 baseline_learnable_pos_emb_dinov2_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch14_dinov2 1369 518"

    # "42 baseline_dinov3_block12 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512"
    # "42 baseline_dinov3_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch16_dinov3 1024 512"
    # "42 baseline_learnable_pos_emb_dinov3_block12 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512"
    # "42 baseline_learnable_pos_emb_dinov3_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch16_dinov3 1024 512"

    # # movi_e
    # "42 gcv2_dinov2_block12 GreedyFeatureInit per_frame networks.HungarianPredictor false ['vit_block12','vit_block_keys12'] vit_block12 vit_base_patch14_dinov2 576 336"
    "42 gcv2_dinov2_output GreedyFeatureInit per_frame networks.HungarianPredictor false ['vit_output','vit_block_keys12'] vit_output vit_base_patch14_dinov2 576 336"
    # "42 gcv2_learnable_pos_emb_dinov2_block12 GreedyFeatureInit per_frame networks.HungarianPredictor true ['vit_block12','vit_block_keys12'] vit_block12 vit_base_patch14_dinov2 576 336"
    "42 gcv2_learnable_pos_emb_dinov2_output GreedyFeatureInit per_frame networks.HungarianPredictor true ['vit_output','vit_block_keys12'] vit_output vit_base_patch14_dinov2 576 336"

    # "42 gcv2_dinov3_block12 GreedyFeatureInit per_frame networks.HungarianPredictor false ['vit_block12','vit_block_keys12'] vit_block12 vit_base_patch16_dinov3 441 336"
    "42 gcv2_dinov3_output GreedyFeatureInit per_frame networks.HungarianPredictor false ['vit_output','vit_block_keys12'] vit_output vit_base_patch16_dinov3 441 336"
    # "42 gcv2_learnable_pos_emb_dinov3_block12 GreedyFeatureInit per_frame networks.HungarianPredictor true ['vit_block12','vit_block_keys12'] vit_block12 vit_base_patch16_dinov3 441 336"
    "42 gcv2_learnable_pos_emb_dinov3_output GreedyFeatureInit per_frame networks.HungarianPredictor true ['vit_output','vit_block_keys12'] vit_output vit_base_patch16_dinov3 441 336"
    
    # baseline
    # "42 baseline_dinov2_block12 FixedLearnedInit first_frame networks.TransformerEncoder false ['vit_block12','vit_block_keys12'] vit_block12 vit_base_patch14_dinov2 576 336"
    "42 baseline_dinov2_output FixedLearnedInit first_frame networks.TransformerEncoder false ['vit_output','vit_block_keys12'] vit_output vit_base_patch14_dinov2 576 336"
    # "42 baseline_learnable_pos_emb_dinov2_block12 FixedLearnedInit first_frame networks.TransformerEncoder true ['vit_block12','vit_block_keys12'] vit_block12 vit_base_patch14_dinov2 576 336"
    "42 baseline_learnable_pos_emb_dinov2_output FixedLearnedInit first_frame networks.TransformerEncoder true ['vit_output','vit_block_keys12'] vit_output vit_base_patch14_dinov2 576 336"

    # "42 baseline_dinov3_block12 FixedLearnedInit first_frame networks.TransformerEncoder false ['vit_block12','vit_block_keys12'] vit_block12 vit_base_patch16_dinov3 441 336"
    "42 baseline_dinov3_output FixedLearnedInit first_frame networks.TransformerEncoder false ['vit_output','vit_block_keys12'] vit_output vit_base_patch16_dinov3 441 336"
    # "42 baseline_learnable_pos_emb_dinov3_block12 FixedLearnedInit first_frame networks.TransformerEncoder true ['vit_block12','vit_block_keys12'] vit_block12 vit_base_patch16_dinov3 441 336"
    "42 baseline_learnable_pos_emb_dinov3_output FixedLearnedInit first_frame networks.TransformerEncoder true ['vit_output','vit_block_keys12'] vit_output vit_base_patch16_dinov3 441 336"
    
    )
    
for config in "${CONFIGS[@]}"; do
    read -r SEED exp_name init_name init_mode predictor use_pos_embed features main_features_key DINO_MODEL NUM_PATCHES input_size<<< "$config"

    echo "Submitting: $exp_name with config: $CONFIG_FILE"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh "${CONFIG_FILE}" \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.initializer.name=${init_name}" \
        "model.initializer.init_mode=${init_mode}" \
        "model.predictor.name=${predictor}" \
        "model.encoder.use_pos_embed=${use_pos_embed}" \
        "model.encoder.backbone.features=${features}" \
        "model.encoder.main_features_key=${main_features_key}" \
        "globals.DINO_MODEL=${DINO_MODEL}" \
        "globals.NUM_PATCHES=${NUM_PATCHES}" \
        "dataset.train_pipeline.transforms.input_size=${input_size}" \
        "dataset.val_pipeline.transforms.input_size=${input_size}" \
        "seed=${SEED}"
done

echo "All jobs submitted!"
