"""
Test the best pre-trained new method v.4 model with test trajectory data.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ae_utilities as aeu
import abnormal_data_generation as adg
import numpy as np
import os
from sklearn.utils.testing import assert_almost_equal

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

trained_model_path = 'model/new_method_v4_5/'
trained_model_summary_results_filename = 'results/new_method_v4_5/summary_results.csv'
detect_summary_filename = trained_model_summary_results_filename[:trained_model_summary_results_filename.rfind('/')] \
                          + '/detect_summary.log'

# Ref.: https://stackoverflow.com/questions/29451030/why-doesnt-np-genfromtxt-remove-header-while-importing-in-python
with open(trained_model_summary_results_filename, 'r') as results:
    line = results.readline()
    header = [e for e in line.strip().split(',') if e]

    results_array = np.genfromtxt(results, names=header, dtype=None, delimiter=',')

# Result arrays
all_n_acc = []
all_a_acc = []
#all_ae_n_acc = []
#all_ae_a_acc = []

for i in range(results_array.shape[0]):
    print('======================== Iteration {} ========================'.format(i))

    mv4 = aeu.BuildOurMethodV4(original_dim=dataset.shape[1],
                               hidden_units=best_layer_type,
                               model_dir_path=trained_model_path,
                               iteration_number=i)

    mv4.load_weights()

    # Test the model and the autoencoder separately
    n_loss, n_acc = mv4.test_model(test_data=dataset)
    a_loss, a_acc = mv4.test_model(test_data=abnormal_data, is_abnormal=True)

    # Refer to the deep_ae_summary_results.csv
    a_loss_ref = results_array['t_loss'][i]
    a_acc_ref = results_array['t_acc'][i]

    # Assert if not approx equal
    assert_almost_equal(a_loss, a_loss_ref, err_msg='Abnormal data loss mismatch.')
    assert_almost_equal(a_acc, a_acc_ref, err_msg='Abnormal data accuracy mismatch.')

    # Store
    all_n_acc.append(n_acc)
    all_a_acc.append(a_acc)

    output_string = '{0:.2f}% of normal samples are detected as normal.\n'.format(n_acc*100.0)
    output_string += '{0:.2f}% of abnormal samples are detected as abnormal.\n'.format(a_acc*100.0)

    print(output_string)
    print('==============================================================')

all_n_acc = np.array(all_n_acc)
all_a_acc = np.array(all_a_acc)

# The best model is the one that as the maximum sum of ratios
best_index = np.argmax(all_n_acc + all_a_acc)

best_trained_model_iteration = int(results_array['iteration'][best_index])
best_n_acc = all_n_acc[best_index]
best_a_acc = all_a_acc[best_index]

# Detect summary
detect_summary_file = open(detect_summary_filename, 'w')

output_string = 'Detect summary of abnormal trajectory detection with deep autoencoder\n'
output_string += '---------------------------------------------------------------------\n'

output_string += '\nThe best one is model_{}:\n'.format(best_index)
output_string += '{0:.2f}% of normal samples are detected as normal.\n'.format(best_n_acc*100.0)
output_string += '{0:.2f}% of abnormal samples are detected as abnormal.\n'.format(best_a_acc*100.0)

detect_summary_file.write(output_string)
print(output_string)

detect_summary_file.close()
