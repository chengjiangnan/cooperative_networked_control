source get_model_specific_info.sh

Z_DIMS="1,2,3,4,5,6,7,8,9"

SAMPLE_IDX=1
TIME_IDX=30

python3 ../ML_functions/visualize_forecasts.py \
--model_name ${MODEL_NAME} \
--forecaster_name ${FORECASTER_NAME} \
--forecaster_hidden_dim ${FORECASTER_HIDDEN_DIM} \
--train_types ${TRAIN_TYPES} \
--z_dims ${Z_DIMS} \
--sample_idx ${SAMPLE_IDX} \
--time_idx ${TIME_IDX} \
--fig_ext ${FIG_EXT}
