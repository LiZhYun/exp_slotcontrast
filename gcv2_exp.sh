#!/bin/bash

# ytvis2021.yaml, movi_c.yaml, movi_d.yaml, movi_e.yaml
CONFIG_FILE="configs/slotcontrast/ytvis2021.yaml"


CONFIGS=(
    # # ytvis2021
    # DINO v2-base
    "42 gcv2_dinov2_block12 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518 768 false"
    "42 gcv2_dinov2_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch14_dinov2 1369 518 768 false"
    "42 gcv2_learnable_pos_emb_dinov2_block12 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518 768 false"
    "42 gcv2_learnable_pos_emb_dinov2_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output  vit_output vit_base_patch14_dinov2 1369 518 768 false"
    "42 gcv2_dinov2_block12_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518 768 true"
    "42 gcv2_dinov2_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch14_dinov2 1369 518 768 true"
    "42 gcv2_learnable_pos_emb_dinov2_block12_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518 768 true"
    "42 gcv2_learnable_pos_emb_dinov2_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output  vit_output vit_base_patch14_dinov2 1369 518 768 true"
    # DINO v2-large
    "42 gcv2_learnable_pos_emb_dinov2_large_block18 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 gcv2_dinov2_large_block18 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 gcv2_learnable_pos_emb_dinov2_large_block24 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 gcv2_dinov2_large_block24 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 gcv2_learnable_pos_emb_dinov2_large_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 gcv2_dinov2_large_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 gcv2_learnable_pos_emb_dinov2_large_block18_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 gcv2_dinov2_large_block18_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 gcv2_learnable_pos_emb_dinov2_large_block24_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 gcv2_dinov2_large_block24_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 gcv2_learnable_pos_emb_dinov2_large_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 gcv2_dinov2_large_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    # DINO v3-base
    "42 gcv2_dinov3_block12 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512 768 false"
    "42 gcv2_dinov3_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch16_dinov3 1024 512 768 false"
    "42 gcv2_learnable_pos_emb_dinov3_block12 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512 768 false"
    "42 gcv2_learnable_pos_emb_dinov3_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_base_patch16_dinov3 1024 512 768 false"
    "42 gcv2_dinov3_block12_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512 768 true"
    "42 gcv2_dinov3_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch16_dinov3 1024 512 768 true"
    "42 gcv2_learnable_pos_emb_dinov3_block12_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512 768 true"
    "42 gcv2_learnable_pos_emb_dinov3_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_base_patch16_dinov3 1024 512 768 true"
    # DINO v3-large
    "42 gcv2_learnable_pos_emb_dinov3_large_block18 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 gcv2_dinov3_large_block18 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 gcv2_learnable_pos_emb_dinov3_large_block24 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 gcv2_dinov3_large_block24 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 gcv2_learnable_pos_emb_dinov3_large_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 gcv2_dinov3_large_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 gcv2_learnable_pos_emb_dinov3_large_block18_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 gcv2_dinov3_large_block18_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 gcv2_learnable_pos_emb_dinov3_large_block24_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 gcv2_dinov3_large_block24_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 gcv2_learnable_pos_emb_dinov3_large_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 gcv2_dinov3_large_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"

    # baseline
    # DINO v2-base
    "42 baseline_dinov2_block12 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518 768 false"
    "42 baseline_dinov2_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch14_dinov2 1369 518 768 false"
    "42 baseline_learnable_pos_emb_dinov2_block12 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518 768 false"
    "42 baseline_learnable_pos_emb_dinov2_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch14_dinov2 1369 518 768 false"
    "42 baseline_dinov2_block12_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518 768 true"
    "42 baseline_dinov2_output_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch14_dinov2 1369 518 768 true"
    "42 baseline_learnable_pos_emb_dinov2_block12_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch14_dinov2 1369 518 768 true"
    "42 baseline_learnable_pos_emb_dinov2_output_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch14_dinov2 1369 518 768 true"
    # DINO v2-large
    "42 baseline_dinov2_large_block18 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 baseline_dinov2_large_block24 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 baseline_dinov2_large_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 baseline_learnable_pos_emb_dinov2_large_block18 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 baseline_learnable_pos_emb_dinov2_large_block24 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 baseline_learnable_pos_emb_dinov2_large_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_large_patch14_dinov2.lvd142m 1369 518 1024 false"
    "42 baseline_dinov2_large_block18_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 baseline_dinov2_large_block24_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 baseline_dinov2_large_output_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 baseline_learnable_pos_emb_dinov2_large_block18_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 baseline_learnable_pos_emb_dinov2_large_block24_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"
    "42 baseline_learnable_pos_emb_dinov2_large_output_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_large_patch14_dinov2.lvd142m 1369 518 1024 true"

    # DINO v3-base
    "42 baseline_dinov3_block12 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512 768 false"
    "42 baseline_dinov3_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch16_dinov3 1024 512 768 false"
    "42 baseline_learnable_pos_emb_dinov3_block12 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512 768 false"
    "42 baseline_learnable_pos_emb_dinov3_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch16_dinov3 1024 512 768 false"
    "42 baseline_dinov3_block12_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512 768 true"
    "42 baseline_dinov3_output_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch16_dinov3 1024 512 768 true"
    "42 baseline_learnable_pos_emb_dinov3_block12_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch16_dinov3 1024 512 768 true"
    "42 baseline_learnable_pos_emb_dinov3_output_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch16_dinov3 1024 512 768 true"
    # DINO v3-large
    "42 baseline_dinov3_large_block18 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 baseline_dinov3_large_block24 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 baseline_dinov3_large_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 baseline_learnable_pos_emb_dinov3_large_block18 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 baseline_learnable_pos_emb_dinov3_large_block24 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 baseline_learnable_pos_emb_dinov3_large_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_large_patch16_dinov3.lvd1689m 1024 512 1024 false"
    "42 baseline_dinov3_large_block18_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 baseline_dinov3_large_block24_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 baseline_dinov3_large_output_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 baseline_learnable_pos_emb_dinov3_large_block18_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 baseline_learnable_pos_emb_dinov3_large_block24_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    "42 baseline_learnable_pos_emb_dinov3_large_output_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_large_patch16_dinov3.lvd1689m 1024 512 1024 true"
    
    
    # # movi_e
    # # DINO v2-base
    # "42 gcv2_dinov2_block12 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch14_dinov2 576 336 768 false"
    # "42 gcv2_dinov2_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch14_dinov2 576 336 768 false"
    # "42 gcv2_learnable_pos_emb_dinov2_block12 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch14_dinov2 576 336 768 false"
    # "42 gcv2_learnable_pos_emb_dinov2_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output  vit_output vit_base_patch14_dinov2 576 336 768 false"
    # "42 gcv2_dinov2_block12_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch14_dinov2 576 336 768 true"
    # "42 gcv2_dinov2_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch14_dinov2 576 336 768 true"
    # "42 gcv2_learnable_pos_emb_dinov2_block12_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch14_dinov2 576 336 768 true"
    # "42 gcv2_learnable_pos_emb_dinov2_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output  vit_output vit_base_patch14_dinov2 576 336 768 true"
    # # DINO v2-large
    # "42 gcv2_learnable_pos_emb_dinov2_large_block18 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 gcv2_dinov2_large_block18 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 gcv2_learnable_pos_emb_dinov2_large_block24 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 gcv2_dinov2_large_block24 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 gcv2_learnable_pos_emb_dinov2_large_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 gcv2_dinov2_large_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 gcv2_learnable_pos_emb_dinov2_large_block18_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 gcv2_dinov2_large_block18_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 gcv2_learnable_pos_emb_dinov2_large_block24_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 gcv2_dinov2_large_block24_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 gcv2_learnable_pos_emb_dinov2_large_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 gcv2_dinov2_large_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # # DINO v3-base
    # "42 gcv2_dinov3_block12 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch16_dinov3 441 336 768 false"
    # "42 gcv2_dinov3_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch16_dinov3 441 336 768 false"
    # "42 gcv2_learnable_pos_emb_dinov3_block12 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch16_dinov3 441 336 768 false"
    # "42 gcv2_learnable_pos_emb_dinov3_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_base_patch16_dinov3 441 336 768 false"
    # "42 gcv2_dinov3_block12_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block12 vit_block12 vit_base_patch16_dinov3 441 336 768 true"
    # "42 gcv2_dinov3_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_base_patch16_dinov3 441 336 768 true"
    # "42 gcv2_learnable_pos_emb_dinov3_block12_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block12 vit_block12 vit_base_patch16_dinov3 441 336 768 true"
    # "42 gcv2_learnable_pos_emb_dinov3_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_base_patch16_dinov3 441 336 768 true"
    # # DINO v3-large
    # "42 gcv2_learnable_pos_emb_dinov3_large_block18 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 gcv2_dinov3_large_block18 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 gcv2_learnable_pos_emb_dinov3_large_block24 GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 gcv2_dinov3_large_block24 GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 gcv2_learnable_pos_emb_dinov3_large_output GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 gcv2_dinov3_large_output GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 gcv2_learnable_pos_emb_dinov3_large_block18_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 gcv2_dinov3_large_block18_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 gcv2_learnable_pos_emb_dinov3_large_block24_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 gcv2_dinov3_large_block24_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 gcv2_learnable_pos_emb_dinov3_large_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor true vit_output vit_output vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 gcv2_dinov3_large_output_norm GreedyFeatureInit per_frame networks.HungarianPredictor false vit_output vit_output vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"

    # # baseline
    # # DINO v2-base
    # "42 baseline_dinov2_block12 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch14_dinov2 576 336 768 false"
    # "42 baseline_dinov2_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch14_dinov2 576 336 768 false"
    # "42 baseline_learnable_pos_emb_dinov2_block12 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch14_dinov2 576 336 768 false"
    # "42 baseline_learnable_pos_emb_dinov2_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch14_dinov2 576 336 768 false"
    # "42 baseline_dinov2_block12_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch14_dinov2 576 336 768 true"
    # "42 baseline_dinov2_output_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch14_dinov2 576 336 768 true"
    # "42 baseline_learnable_pos_emb_dinov2_block12_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch14_dinov2 576 336 768 true"
    # "42 baseline_learnable_pos_emb_dinov2_output_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch14_dinov2 576 336 768 true"
    # # DINO v2-large
    # "42 baseline_dinov2_large_block18 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 baseline_dinov2_large_block24 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 baseline_dinov2_large_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 baseline_learnable_pos_emb_dinov2_large_block18 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 baseline_learnable_pos_emb_dinov2_large_block24 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 baseline_learnable_pos_emb_dinov2_large_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_large_patch14_dinov2.lvd142m 576 336 1024 false"
    # "42 baseline_dinov2_large_block18_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 baseline_dinov2_large_block24_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 baseline_dinov2_large_output_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 baseline_learnable_pos_emb_dinov2_large_block18_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block18 vit_block18 vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 baseline_learnable_pos_emb_dinov2_large_block24_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block24 vit_block24 vit_large_patch14_dinov2.lvd142m 576 336 1024 true"
    # "42 baseline_learnable_pos_emb_dinov2_large_output_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_large_patch14_dinov2.lvd142m 576 336 1024 true"

    # # DINO v3-base
    # "42 baseline_dinov3_block12 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch16_dinov3 441 336 768 false"
    # "42 baseline_dinov3_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch16_dinov3 441 336 768 false"
    # "42 baseline_learnable_pos_emb_dinov3_block12 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch16_dinov3 441 336 768 false"
    # "42 baseline_learnable_pos_emb_dinov3_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch16_dinov3 441 336 768 false"
    # "42 baseline_dinov3_block12_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block12 vit_block12 vit_base_patch16_dinov3 441 336 768 true"
    # "42 baseline_dinov3_output_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_base_patch16_dinov3 441 336 768 true"
    # "42 baseline_learnable_pos_emb_dinov3_block12_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block12 vit_block12 vit_base_patch16_dinov3 441 336 768 true"
    # "42 baseline_learnable_pos_emb_dinov3_output_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_base_patch16_dinov3 441 336 768 true"
    # # DINO v3-large
    # "42 baseline_dinov3_large_block18 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 baseline_dinov3_large_block24 FixedLearnedInit first_frame networks.TransformerEncoder false vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 baseline_dinov3_large_output FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 baseline_learnable_pos_emb_dinov3_large_block18 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 baseline_learnable_pos_emb_dinov3_large_block24 FixedLearnedInit first_frame networks.TransformerEncoder true vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 baseline_learnable_pos_emb_dinov3_large_output FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_large_patch16_dinov3.lvd1689m 441 336 1024 false"
    # "42 baseline_dinov3_large_block18_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 baseline_dinov3_large_block24_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 baseline_dinov3_large_output_norm FixedLearnedInit first_frame networks.TransformerEncoder false vit_output vit_output vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 baseline_learnable_pos_emb_dinov3_large_block18_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block18 vit_block18 vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 baseline_learnable_pos_emb_dinov3_large_block24_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_block24 vit_block24 vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"
    # "42 baseline_learnable_pos_emb_dinov3_large_output_norm FixedLearnedInit first_frame networks.TransformerEncoder true vit_output vit_output vit_large_patch16_dinov3.lvd1689m 441 336 1024 true"

    )
    
for config in "${CONFIGS[@]}"; do
    read -r SEED exp_name init_name init_mode predictor use_pos_embed features main_features_key DINO_MODEL NUM_PATCHES input_size FEAT_DIM normalize_features <<< "$config"

    echo "Submitting: $exp_name with config: $CONFIG_FILE"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh "${CONFIG_FILE}" \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.initializer.name=${init_name}" \
        "model.initializer.init_mode=${init_mode}" \
        "model.predictor.name=${predictor}" \
        "model.encoder.use_pos_embed=${use_pos_embed}" \
        "model.encoder.backbone.features=${features}" \
        "model.encoder.main_features_key=${main_features_key}" \
        "model.encoder.normalize_features=${normalize_features}" \
        "globals.DINO_MODEL=${DINO_MODEL}" \
        "globals.NUM_PATCHES=${NUM_PATCHES}" \
        "globals.FEAT_DIM=${FEAT_DIM}" \
        "dataset.train_pipeline.transforms.input_size=${input_size}" \
        "dataset.val_pipeline.transforms.input_size=${input_size}" \
        "seed=${SEED}"
done

echo "All jobs submitted!"
