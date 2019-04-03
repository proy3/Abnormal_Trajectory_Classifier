"""
Train Abnormal trajectory detection with Isolation Forest model.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ae_utilities as aeu
import input_data as data
import abnormal_data_generation as adg
import dataset_defines as dd
import numpy as np
import os
import time


trad_method_name = 'Isolation Forest'

# Extract trajectories and export data to array
dataset_file = data.extract_augment_and_export_data(raw_input_file_all=dd.raw_input_file_all,
                                                    input_raw_image_frame_path=dd.input_raw_image_frame_path,
                                                    raw_input_file_names=dd.raw_input_file_names,
                                                    video_data_fps=dd.video_data_fps,
                                                    generate_graph=False,
                                                    show_graph=False)

dataset = np.genfromtxt(os.path.join(data.dir_name, dataset_file), delimiter=',')

# Ignore first column representing object_id
dataset = dataset[:,1:]

# Generate abnormal data
abnormal_data = adg.generate_abnormal_data(n_objects=20, generate_graph=False, show_graph=False)
abnormal_data = abnormal_data[:,1:]

# Files setup
test_score_filename = 'results/isolation_forest/test_scores.csv'
summary_results_filename = test_score_filename[:test_score_filename.rfind('/')] + '/summary_results.csv'
global_summary_filename = test_score_filename[:test_score_filename.rfind('/')] + '/global_summary.log'
model_files_dir_name = 'model/isolation_forest/'

data.make_dir_if_new(test_score_filename)
data.make_dir_if_new(model_files_dir_name)

normal_train_ratio_list = []
normal_valid_ratio_list = []
abnormal_ratio_list = []

for i in range(aeu.repeat_number):
    print('======================== Iteration {} ========================'.format(i))
    # Shuffle the data by row only
    # and get the seed in order to reproduce the random sequence
    train_data, validation_data, random_shuffle_seed = data.split_dataset_uniformly(dataset)

    # The trained model will be saved
    saved_ae_network_path = os.path.join(data.dir_name, model_files_dir_name)

    svm = aeu.BuildSimpleOutliersDetectionMethod(trad_method_name)

    svm.train(train_data=train_data,
              save_model=True,
              test_saved_model=True,
              model_dir_path=saved_ae_network_path,
              iteration_number=i)

    # Get the scores
    svm_train_acc = svm.accuracy
    svm_train_predicted_class = svm.predicted_class

    svm_validate_acc, svm_validate_predicted_class = aeu.test_trained_traditional_model(test_data=validation_data,
                                                                                        clf_name=trad_method_name,
                                                                                        model_dir_path=saved_ae_network_path,
                                                                                        iteration_number=i)

    svm_tests_acc, svm_tests_predicted_class = aeu.test_trained_traditional_model(test_data=abnormal_data,
                                                                                  clf_name=trad_method_name,
                                                                                  model_dir_path=saved_ae_network_path,
                                                                                  iteration_number=i,
                                                                                  is_abnormal=True)

    # Format the test scores: [Iteration, Test_sample_No, Score]
    if i == 0:
        with open(os.path.join(data.dir_name, test_score_filename), 'wb') as score_file:
            score_file.write(b'iteration,test_sample_no,predicted_class\n')

    rows_number = len(svm_tests_predicted_class)
    test_scores_array = np.hstack((np.full((rows_number, 1), i), np.array(range(rows_number)).reshape(-1, 1),
                                   np.array(svm_tests_predicted_class).reshape(-1, 1)))

    # Export the test scores to text file
    with open(os.path.join(data.dir_name, test_score_filename), 'ab') as score_file:
        np.savetxt(score_file, test_scores_array, delimiter=',')

    output_string = 'Iteration {}: svm_train_acc = {}; svm_validation_acc = {}'\
        .format(i, svm_train_acc, svm_validate_acc)

    print('\n')

    # Save the result to a global summary file
    output_string += '\n'

    # Summary file format: [Iteration, ae_train_score, ae_validate_score, threshold_value,
    #                       normal_train_ratio, normal_valid_ratio, abnormal_ratio]
    if i == 0:
        with open(os.path.join(data.dir_name, summary_results_filename), 'wb') as summary_file:
            summary_file.write(b'iteration,random_shuffle_seed,normal_train_ratio,normal_valid_ratio,abnormal_ratio\n')

    normal_train_ratio_list.append(svm_train_acc*100.0)
    normal_valid_ratio_list.append(svm_validate_acc*100.0)
    abnormal_ratio_list.append(svm_tests_acc*100.0)

    with open(os.path.join(data.dir_name, summary_results_filename), 'ab') as summary_file:
        np.savetxt(summary_file, np.array([i,random_shuffle_seed,svm_train_acc,svm_validate_acc,
                                           svm_tests_acc]).reshape(1, -1), delimiter=',')

    output_string += '{0:.2f}% of normal training samples are detected as normal.\n'.format(svm_train_acc*100.0)
    output_string += '{0:.2f}% of normal validation samples are detected as normal.\n'.format(svm_validate_acc*100.0)
    output_string += '{0:.2f}% of abnormal samples are detected as abnormal.\n'.format(svm_tests_acc*100.0)

    print(output_string)
    print('==============================================================')

# Global summary
global_summary_file = open(global_summary_filename, 'w')

output_string = 'Global summary of abnormal trajectory detection using SVM Classifier model\n'
output_string += '--------------------------------------------------------------------------\n'
output_string += '\t{0:.2f}% of normal training samples are detected as normal;\n'.format(np.mean(
    normal_train_ratio_list))
output_string += '\t{0:.2f}% of normal validation samples are detected as normal;\n'.format(np.mean(
    normal_valid_ratio_list))
output_string += '\t{0:.2f}% of abnormal samples are detected as abnormal.\n'.format(np.mean(abnormal_ratio_list))

global_summary_file.write(output_string)
print(output_string)

global_summary_file.close()
