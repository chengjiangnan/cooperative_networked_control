source get_model_specific_info.sh

python3 ../PCA_functions/pca_compute_stats.py \
--model_name ${MODEL_NAME} \
--train_types ${TRAIN_TYPES} \
--allowed_error_per ${ALLOWED_ERROR_PER}
