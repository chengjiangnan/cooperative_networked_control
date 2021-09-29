# Cooperative Networked Control #
This is the code release for our NeurIPS 2021 paper "Data Sharing and Compression for Cooperative Networked Control".

### Setup ###

* export `ROBOTICS_CODESIGN_DIR` in your bashrc to be the path of this repo
    * example `export ROBOTICS_CODESIGN_DIR=<path>/`

* have `python3` installed. My version is 3.7.4

* have key packages installed. Including `qpth`

### Models ###

There are 5 subdirectories for evaluation under `./MPC_control/`: 

* `eval_pca_full`, `eval_pca_mpc` corresponding to our low-rank approximation (equivalent to PCA) experiments based on one-shot problem (Fig.5) and mpc problem (Fig.6), respectively;

* `eval_box_iot`, `eval_biased_cell`, `eval_pjm`, corresponding to our IoT, taxi scheduling and battery charing scenarios, respectively.

### Command ###

* Low-rank approximation (PCA) on the synthetic data (Take `pca_full` as an example)
```
cd ./MPC_control/eval_pca_full
./pca_generate_data.sh // Generate data
./pca_train.sh // Training
./pca_joint_plot.sh // Plot main results (Fig. 5(a)-(c))
```
Then you can check the resulted figures under the `./scratch/` folder. We also have a few scripts for result visualization/statistics, including:
```
./pca_joint_plot_2.sh // Combine two simulation results together, like Fig. 5(d)
./pca_compute_stats.sh // Output the statistics
```

* Task-aware training on the real data (Take `pjm` as an example)
```
cd ./MPC_control/eval_pjm
./pjm_generate_data.sh // Read raw data
./pjm_train.sh // Training
./pjm_joint_plot.sh // Plot main results (Fig. 2, first row)
```
Then you can check the resulted figures under the `./scratch/` folder. We also have a few scripts for result visualization/statistics, including:
```
./pjm_visualize_data.sh // Visualize the training and testing data
./pjm_visualize_forecast_errors.sh // Visualize the forecasting errors from a few different angles (Fig. 2, second and third rows, & Fig. 10)
./pjm_visualize_forecasts.sh // Visualize the time-domain forecasts (Fig. 14)
./pjm_visualize_state_evolution.sh // Visualize the state evolution (Fig. 15)
./pjm_visualize_train_loss_evolution.sh // Visualize the train loss evolution
./pjm_compute_stats.sh // Output the statistics
```

Note that by default on our machine, `pjm` experiment takes less than 1 hours, while `box_iot` and `biased_cell` experiments take 2~4 days for training, since the later two have to call QP solvers.
