# Abnormal_Trajectory_Classifier
This repository contains the source code of some published articles related to the detection of abnormal trajectories of moving objects in surveillance videos using unsupervised machine learning approaches.

The following sub reposities contains source code for the following papers:
- DAE_Method: contains code for the ISVC'18 article [Road User Abnormal Trajectory Detection using a Deep Autoencoder](https://arxiv.org/abs/1809.00957).
- ALREC_Method: contains code for the CRV'19 article [Adversarially Learned Abnormal Trajectory Classifier](https://arxiv.org/abs/1903.11040v1).


# Library based Dependencies:

- numpy
- sklearn
- keras
- tensorflow
- scipy
- sqlite3
- pandas
- PIL


# Instructions

## DAE_Method

- Note that Each sub folder corresponds to a specific Urban Tracker [dataset](https://www.jpjodoin.com/urbantracker/index.htm) video.

- For a specific Urban Tracker video:
  1. Modify the path of groundtruth files in dataset_defines.py
  2. Train the Deep Autoencoder model by running the train_deep_abnormal_traj_detect_model.py 
  3. Detect abnormal trajectories by running detect_abnormal_traj_with_trained_model.py

## ALREC_Method

- Note that the sub folders rene, rouen, sherbrooke and stmarc correspond to the Urban Tracker [dataset](https://www.jpjodoin.com/urbantracker/index.htm).
- The sub folder caviar_inria corresponds to the INRIA scenario of the CAVIAR Test Case [dataset](http://groups.inf.ed.ac.uk/vision/CAVIAR/CAVIARDATA1/).

- For a specific Urban Tracker video:
  1. Modify the path of groundtruth files in dataset_defines.py
  2. Train the Deep Autoencoder model via ALREC method by running the train_new_method_v4_for_atd.py 
  3. Detect abnormal trajectories by running detect_atd_with_best_new_method_v4.py

- For the INRIA scenario of the CAVIAR Test Case:
  1. Modify the path of groundtruth files in dataset_defines.py
  2. Train the Deep Autoencoder model via ALREC method by running the train_new_method_v4_for_atd.py 
  3. Detect abnormal complete trajectories by running test_alrec_for_complete_atd.py

