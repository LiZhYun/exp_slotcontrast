DATA_DIR="/home/zhiyuan/Codes/POST3R/post3r/data/"
OUTPUT_DIR="/home/zhiyuan/Codes/exp_slotcontrast/logs"

export PYTHONPATH="/home/zhiyuan/Codes/exp_slotcontrast:$PYTHONPATH"

python slotcontrast/train.py "configs/slotcontrast/ytvis2021.yaml" \
        "experiment_name=slotcontrast_gated_attention_grouper_w_contrast" \
        "model.latent_processor.use_ttt3r=false" \
        "model.grouper.use_gated=true" \
        "model.grouper.use_ttt=false" \
        "model.grouper.use_gru=false" \
        "model.loss_weights.loss_ss=0.5" \
        --data-dir ${DATA_DIR} \
        --log-dir ${OUTPUT_DIR} \
