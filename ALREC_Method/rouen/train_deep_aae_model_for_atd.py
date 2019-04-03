"""
Train Abnormal trajectory detection with deep autoencoder.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ae_utilities as aeu
import input_data as data
import abnormal_data_generation as adg
import dataset_defines as dd
import numpy as np
import os


abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
dataset_name = dir_name[dir_name.rfind('/')+1:] + '_gt_data.csv'
dataset_file_path = os.path.join(dir_name + '/data', dataset_name)
abnormal_name = dir_name[dir_name.rfind('/')+1:] + '_gt_real_abnormal_2.csv'
abnormal_file_path = os.path.join(dir_name + '/data', abnormal_name)

# Extract trajectories and export data to array
dataset = np.genfromtxt(dataset_file_path, delimiter=',')

# Ignore first column representing object_id
dataset = dataset[:,1:]

# Generate abnormal data
abnormal_data = np.genfromtxt(abnormal_file_path, delimiter=',')
abnormal_data = abnormal_data[:,1:]

# Best layer type tested in main_test.py
# ref.: https://deeplearning4j.org/deepautoencoder
best_layer_type = (128,64,32,16,8)

# Files setup
test_score_filename = 'results/deep_aae/test_scores.csv'
summary_results_filename = test_score_filename[:test_score_filename.rfind('/')] + '/summary_results.csv'
global_summary_filename = test_score_filename[:test_score_filename.rfind('/')] + '/global_summary.log'
model_files_dir_name = 'model/deep_aae/'

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

    aae = aeu.BuildSimpleAAE(original_dim=train_data.shape[1],
                             hidden_units=best_layer_type,
                             model_dir_path=saved_ae_network_path)

    aae.train(train_data=train_data,
              save_model=True,
              iteration_number=i,
              print_and_plot_history=True,
              show_plots=False)

    # Get the scores
    ae_train_score = aae.global_mse
    ae_train_scores = aae.mse_per_sample

    ae_validate_score, ae_validate_scores = aae.test_model(test_data=validation_data,
                                                           iteration_number=i)

    ae_tests_score, ae_tests_scores = aae.test_model(test_data=abnormal_data,
                                                     iteration_number=i)

    # Format the test scores: [Iteration, Test_sample_No, Score]
    if i == 0:
        with open(os.path.join(data.dir_name, test_score_filename), 'wb') as score_file:
            score_file.write(b'iteration,test_sample_no,score\n')

    rows_number = len(ae_tests_scores)
    test_scores_array = np.hstack((np.full((rows_number, 1), i), np.array(range(rows_number)).reshape(-1, 1),
                                   np.array(ae_tests_scores).reshape(-1, 1)))

    # Export the test scores to text file
    with open(os.path.join(data.dir_name, test_score_filename), 'ab') as score_file:
        np.savetxt(score_file, test_scores_array, delimiter=',')

    output_string = 'Iteration {} with layer type {}: ae_train_score = {}; ae_validation_score = {}'\
        .format(i, best_layer_type, ae_train_score, ae_validate_score)

    print('\n')

    # Save the result to a global summary file
    output_string += '\n'

    threshold_value = ae_train_score + 3 * np.std(ae_train_scores)

    # Summary file format: [Iteration, ae_train_score, ae_validate_score, threshold_value,
    #                       normal_train_ratio, normal_valid_ratio, abnormal_ratio]
    if i == 0:
        with open(os.path.join(data.dir_name, summary_results_filename), 'wb') as summary_file:
            summary_file.write(b'iteration,random_shuffle_seed,ae_train_score,ae_validate_score,threshold_value,'
                               b'normal_train_ratio,normal_valid_ratio,abnormal_ratio\n')

    normal_train_ratio = sum([score < threshold_value for score in ae_train_scores])/float(len(ae_train_scores))
    normal_valid_ratio = sum([score < threshold_value for score in ae_validate_scores])/float(len(ae_validate_scores))
    abnormal_ratio = sum([score > threshold_value for score in ae_tests_scores])/float(len(ae_tests_scores))

    normal_train_ratio_list.append(normal_train_ratio*100.0)
    normal_valid_ratio_list.append(normal_valid_ratio*100.0)
    abnormal_ratio_list.append(abnormal_ratio*100.0)

    with open(os.path.join(data.dir_name, summary_results_filename), 'ab') as summary_file:
        np.savetxt(summary_file, np.array([i,random_shuffle_seed,ae_train_score,ae_validate_score,threshold_value,
                                           normal_train_ratio,normal_valid_ratio,abnormal_ratio]).reshape(1, -1),
                   delimiter=',')

    output_string += '{0:.2f}% of normal training samples are detected as normal.\n'.format(normal_train_ratio*100.0)
    output_string += '{0:.2f}% of normal validation samples are detected as normal.\n'.format(normal_valid_ratio*100.0)
    output_string += '{0:.2f}% of abnormal samples are detected as abnormal.\n'.format(abnormal_ratio*100.0)

    print(output_string)
    print('==============================================================')

# Global summary
global_summary_file = open(global_summary_filename, 'w')

output_string = 'Global summary of abnormal trajectory detection with deep adversarial autoencoder\n'
output_string += '---------------------------------------------------------------------------------\n'
output_string += 'On average, using layer type {},\n'.format(best_layer_type)
output_string += '\t{0:.2f}% of normal training samples are detected as normal;\n'.format(np.mean(
    normal_train_ratio_list))
output_string += '\t{0:.2f}% of normal validation samples are detected as normal;\n'.format(np.mean(
    normal_valid_ratio_list))
output_string += '\t{0:.2f}% of abnormal samples are detected as abnormal.\n'.format(np.mean(abnormal_ratio_list))

global_summary_file.write(output_string)
print(output_string)

global_summary_file.close()