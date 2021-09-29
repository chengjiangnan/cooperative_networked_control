source get_model_specific_info.sh

Z_DIMS="1,2,3,4,5,6,7,8,9"
SAMPLE_IDX=1

python3 ../ML_functions/visualize_state_evolution.py \
--model_name ${MODEL_NAME} \
--train_types ${TRAIN_TYPES} \
--z_dims ${Z_DIMS} \
--sample_idx ${SAMPLE_IDX} \
--fig_ext ${FIG_EXT}
