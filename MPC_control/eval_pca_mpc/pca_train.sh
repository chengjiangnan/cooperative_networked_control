source get_model_specific_info.sh

python3 ../PCA_functions/pca_mpc.py \
--model_name ${MODEL_NAME} \
--controller_name ${CONTROLLER_NAME} \
--train_types ${TRAIN_TYPES} \
--weights ${WEIGHTS} \
--min_z ${MIN_Z} \
--max_z ${MAX_Z}