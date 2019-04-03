"""
Test the pre-trained autoencoder model with test trajectory data.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ae_utilities as aeu
import abnormal_data_generation as adg
import numpy as np

trained_model_path = 'model/deep_ae/'
trained_model_summary_results_filename = 'results/deep_ae/summary_results.csv'
detect_summary_filename = trained_model_summary_results_filename[:trained_model_summary_results_filename.rfind('/')] \
                          + '/detect_summary.log'

# Ref.: https://stackoverflow.com/questions/29451030/why-doesnt-np-genfromtxt-remove-header-while-importing-in-python
with open(trained_model_summary_results_filename, 'r') as results:
    line = results.readline()
    header = [e for e in line.strip().split(',') if e]

    results_array = np.genfromtxt(results, names=header, dtype=None, delimiter=',')

# Generate abnormal data
abnormal_data = adg.generate_abnormal_data_from_raw_data(overwrite_data=False,
                                                         generate_graph=False, show_graph=False)
abnormal_data = abnormal_data[:,1:]

abnormal_ratios = []

for i in range(results_array.shape[0]):
    print('======================== Iteration {} ========================'.format(i))

    ae_tests_score, ae_tests_scores = aeu.test_trained_ae_model(test_data=abnormal_data,
                                                                model_dir_path=trained_model_path,
                                                                iteration_number=i)

    # Refer to the deep_ae_summary_results.csv
    threshold_value = results_array['threshold_value'][i]

    abnormal_ratio = sum([score > threshold_value for score in ae_tests_scores])/float(len(ae_tests_scores))

    abnormal_ratios.append(abnormal_ratio)

    output_string = '\n{0:.2f}% of abnormal samples are detected as abnormal.'.format(abnormal_ratio*100.0)
    print(output_string)
    print('==============================================================')

abnormal_ratios = np.array(abnormal_ratios)

# The best model is the one that as the maximum sum of ratios
best_index = np.argmax(results_array['normal_train_ratio'] +
                       results_array['normal_valid_ratio'] +
                       abnormal_ratios)

best_trained_model_iteration = int(results_array['iteration'][best_index])
best_abnormal_ratio = abnormal_ratios[best_index]

# Detect summary
detect_summary_file = open(detect_summary_filename, 'w')

output_string = 'Detect summary of abnormal trajectory detection with deep autoencoder\n'
output_string += '---------------------------------------------------------------------\n'

output_string += '\nThe best one is model_{} with {:.2f}% of abnormal samples are detected as abnormal.\n'\
    .format(best_trained_model_iteration, best_abnormal_ratio*100.0)

detect_summary_file.write(output_string)
print(output_string)

detect_summary_file.close()
