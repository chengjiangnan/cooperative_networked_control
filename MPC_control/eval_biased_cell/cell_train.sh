source get_model_specific_info.sh

python3 ../ML_functions/train.py \
--model_name ${MODEL_NAME} \
--controller_name ${CONTROLLER_NAME} \
--forecaster_name ${FORECASTER_NAME} \
--forecaster_hidden_dim ${FORECASTER_HIDDEN_DIM} \
--train_types ${TRAIN_TYPES} \
--min_z ${MIN_Z} \
--max_z ${MAX_Z} \
--num_epochs 1000
