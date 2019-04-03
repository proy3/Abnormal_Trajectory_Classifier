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
test_score_filename = 'results/new_method_v4_5/test_scores.csv'
summary_results_filename = test_score_filename[:test_score_filename.rfind('/')] + '/summary_results.csv'
global_summary_filename = test_score_filename[:test_score_filename.rfind('/')] + '/global_summary.log'
model_files_dir_name = 'model/new_method_v4_5/'

data.make_dir_if_new(test_score_filename)
data.make_dir_if_new(model_files_dir_name)

n_acc_list = []
v_acc_list = []
t_acc_list = []
ae_n_acc_list = []
ae_v_acc_list = []
ae_t_acc_list = []

for i in range(aeu.repeat_number):
    print('======================== Iteration {} ========================'.format(i))
    # Shuffle the data by row only
    # and get the seed in order to reproduce the random sequence
    train_data, validation_data, random_shuffle_seed = data.split_dataset_uniformly(dataset)

    # The trained model will be saved
    saved_ae_network_path = os.path.join(data.dir_name, model_files_dir_name)

    mv4 = aeu.BuildOurMethodV4(original_dim=train_data.shape[1],
                               hidden_units=best_layer_type,
                               model_dir_path=saved_ae_network_path,
                               iteration_number=i)

    mv4.train(train_data=train_data,
              save_model=True,
              print_and_plot_history=True,
              show_plots=False)

    n_loss, n_acc, ae_n_mse, ae_n_mses = mv4.test_model(test_data=train_data, test_ae=True)
    v_loss, v_acc, ae_v_mse, ae_v_mses = mv4.test_model(test_data=validation_data, test_ae=True)
    t_loss, t_acc, ae_t_mse, ae_t_mses = mv4.test_model(test_data=abnormal_data, is_abnormal=True, test_ae=True)

    output_string = 'Iteration {} with layer type {}: n_loss = {}; v_loss = {}; t_loss = {}'\
        .format(i, best_layer_type, n_loss, v_loss, t_loss)

    print('\n')

    # Save the result to a global summary file
    output_string += '\n'

    # Compute the threshold value for the autoencoder method. Used for the comparison purpose
    ae_threshold = ae_n_mse + ae_v_mse + 3 * (np.std(ae_n_mses) + np.std(ae_v_mses))

    # Compute the accuracy using the old method: only using the autoencoder with the computed threshold
    ae_n_acc = sum([score < ae_threshold for score in ae_n_mses])/float(len(ae_n_mses))
    ae_v_acc = sum([score < ae_threshold for score in ae_v_mses])/float(len(ae_v_mses))
    ae_t_acc = sum([score > ae_threshold for score in ae_t_mses])/float(len(ae_t_mses))

    # Summary file format: [Iteration, ae_train_score, ae_validate_score, threshold_value,
    #                       normal_train_ratio, normal_valid_ratio, abnormal_ratio]
    if i == 0:
        with open(os.path.join(data.dir_name, summary_results_filename), 'wb') as summary_file:
            summary_file.write(b'iteration,random_shuffle_seed,ae_threshold,ae_n_acc,ae_v_acc,ae_t_acc,'
                               b'n_loss,n_acc,v_loss,v_acc,t_loss,t_acc\n')

    n_acc_list.append(n_acc*100.0)
    v_acc_list.append(v_acc*100.0)
    t_acc_list.append(t_acc*100.0)
    ae_n_acc_list.append(ae_n_acc*100.0)
    ae_v_acc_list.append(ae_v_acc*100.0)
    ae_t_acc_list.append(ae_t_acc*100.0)

    with open(os.path.join(data.dir_name, summary_results_filename), 'ab') as summary_file:
        np.savetxt(summary_file, np.array([i,random_shuffle_seed,ae_threshold,ae_n_acc,ae_v_acc,ae_t_acc,
                                           n_loss,n_acc,v_loss,v_acc,t_loss,t_acc]).reshape(1, -1),delimiter=',')

    output_string += '{:.2f}% (old: {:.2f}%) of normal train samples are detected as normal.\n'.format(n_acc*100.0,
                                                                                                       ae_n_acc*100.0)
    output_string += '{:.2f}% (old: {:.2f}%) of normal valid samples are detected as normal.\n'.format(v_acc*100.0,
                                                                                                       ae_v_acc*100.0)
    output_string += '{:.2f}% (old: {:.2f}%) of abnormal samples are detected as abnormal.\n'.format(t_acc*100.0,
                                                                                                     ae_t_acc*100.0)

    print(output_string)
    print('==============================================================')

# Global summary
global_summary_file = open(global_summary_filename, 'w')

output_string = 'Global summary of abnormal trajectory detection with our new method v.4\n'
output_string += '-----------------------------------------------------------------------\n'
output_string += 'On average, using layer type {},\n'.format(best_layer_type)
output_string += '\t{:.2f}% (old:{:.2f}%) of normal training samples are detected as normal;\n'.format(
    np.mean(n_acc_list), np.mean(ae_n_acc_list))
output_string += '\t{:.2f}% (old:{:.2f}%) of normal validation samples are detected as normal;\n'.format(
    np.mean(v_acc_list), np.mean(ae_v_acc_list))
output_string += '\t{:.2f}% (old:{:.2f}%) of abnormal samples are detected as abnormal.\n'.format(
    np.mean(t_acc_list), np.mean(ae_t_acc_list))
output_string += '-----------------------------------------------------------------------\n'
output_string += 'On maximum, using layer type {},\n'.format(best_layer_type)
output_string += '\t{:.2f}% (old:{:.2f}%) of normal training samples are detected as normal;\n'.format(
    np.max(n_acc_list), np.max(ae_n_acc_list))
output_string += '\t{:.2f}% (old:{:.2f}%) of normal validation samples are detected as normal;\n'.format(
    np.max(v_acc_list), np.max(ae_v_acc_list))
output_string += '\t{:.2f}% (old:{:.2f}%) of abnormal samples are detected as abnormal.\n'.format(
    np.max(t_acc_list), np.max(ae_t_acc_list))
output_string += '-----------------------------------------------------------------------\n'
output_string += 'On minimum, using layer type {},\n'.format(best_layer_type)
output_string += '\t{:.2f}% (old:{:.2f}%) of normal training samples are detected as normal;\n'.format(
    np.min(n_acc_list), np.min(ae_n_acc_list))
output_string += '\t{:.2f}% (old:{:.2f}%) of normal validation samples are detected as normal;\n'.format(
    np.min(v_acc_list), np.min(ae_v_acc_list))
output_string += '\t{:.2f}% (old:{:.2f}%) of abnormal samples are detected as abnormal.\n'.format(
    np.min(t_acc_list), np.min(ae_t_acc_list))
output_string += '-----------------------------------------------------------------------\n'

global_summary_file.write(output_string)
print(output_string)

global_summary_file.close()
