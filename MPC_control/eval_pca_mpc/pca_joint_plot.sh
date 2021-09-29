source get_model_specific_info.sh

python3 ../PCA_functions/pca_joint_plot.py \
--model_name ${MODEL_NAME} \
--train_types ${TRAIN_TYPES} \
--weights ${WEIGHTS} \
--fig_ext ${FIG_EXT}