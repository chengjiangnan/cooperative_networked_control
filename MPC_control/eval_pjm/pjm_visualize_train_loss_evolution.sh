source get_model_specific_info.sh

Z_DIM=3
TRAIN_TYPE=task_agnostic
FREQ=10

python3 ../ML_functions/visualize_train_loss_evolution.py \
--model_name ${MODEL_NAME} \
--train_type ${TRAIN_TYPE} \
--z_dim ${Z_DIM} \
--freq ${FREQ} \
--fig_ext ${FIG_EXT}
