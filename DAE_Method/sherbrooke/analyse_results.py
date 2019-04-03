"""
Test the pre-trained autoencoder model with test trajectory data.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ae_utilities as aeu
import dataset_defines as dd
import numpy as np
import os


abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
dataset_name = dir_name[dir_name.rfind('/')+1:] + '_gt_data.csv'
dataset_file_path = os.path.join(dir_name + '/data', dataset_name)
abnormal_name = dir_name[dir_name.rfind('/')+1:] + '_gt_real_abnormal_2.csv'
abnormal_file_path = os.path.join(dir_name + '/data', abnormal_name)


def get_comparison_results(result_file):
    """
    Returns a array of strings in the following format:
    [[label_0, Size_N, size_A, TPR, TNR, TPR, TNR, TPR, TNR, TPR, TNR],
     [label_1, Size_N, size_A, TPR, TNR, TPR, TNR, TPR, TNR, TPR, TNR],
                                ...
     [total, Size_N, size_A, TPR, TNR, TPR, TNR, TPR, TNR, TPR, TNR]]
    """
    # Change the directory
    or_dir_name = os.getcwd()
    os.chdir(dir_name)

    list_of_model_names = ['one_class_svm','isolation_forest','single_ae','deep_ae']

    # Extract trajectories and export data to array
    dataset = np.genfromtxt(dataset_file_path, delimiter=',')

    # Ignore first column representing object_id
    dataset = dataset[:,1:]

    # Generate abnormal data
    abnormal_data = np.genfromtxt(abnormal_file_path, delimiter=',')
    abnormal_data = abnormal_data[:,1:]

    # list of object labels: -1 means all objects. The following order follows: 1. Cars; 2. Peds; 3. Bike; 4.All
    list_labels = [1,0,2,3]
    label_names = ['Peds','Cars','Bike','All']

    # Get the number of labels
    n_labels = 1
    for object_label in list_labels:
        if object_label != 3:
            if len(dataset[dataset[:,0] == object_label]) > 0:
                n_labels += 1

    row_string = r'\multirow{{{}}}{{*}}{{Sherb.}}'.format(n_labels)

    is_first = True

    for object_label in list_labels:
        print('====================================== {} ======================================'.
              format(label_names[object_label]))

        sub_normal_data = dataset
        sub_abnormal_data = abnormal_data
        if object_label != 3:
            sub_normal_data = sub_normal_data[sub_normal_data[:,0] == object_label]
            sub_abnormal_data = sub_abnormal_data[sub_abnormal_data[:,0] == object_label]
            if len(sub_normal_data) == 0 or len(sub_abnormal_data) == 0:
                continue

        if is_first:
            result_file.write(r'\multicolumn{{1}}{{c|}}{{{}}} & '.format(row_string))
            is_first = False
        else:
            result_file.write(r'\multicolumn{1}{c|}{} & ')

        # Get the number of samples
        size_n = sub_normal_data.shape[0]
        size_a = sub_abnormal_data.shape[0]

        result_file.write(r'{} & {} & {} '.format(label_names[object_label], size_n, size_a))

        for model_name in list_of_model_names:
            print('================================== {} =================================='.format(model_name))
            # Files containing info of the model and threshold value
            trained_model_path = 'model/' + model_name + '/'
            trained_model_summary_results_filename = 'results/' + model_name + '/summary_results.csv'

            # Ref.: https://stackoverflow.com/questions/29451030/why-doesnt-np-genfromtxt-remove-header-while-importing-in-python
            with open(trained_model_summary_results_filename, 'r') as results:
                line = results.readline()
                header = [e for e in line.strip().split(',') if e]
                results_array = np.genfromtxt(results, names=header, dtype=None, delimiter=',')

            TPR_list = []
            TNR_list = []

            for i in range(aeu.repeat_number):
                print('======================== Iteration {} ========================'.format(i))

                if model_name == 'single_ae' or model_name == 'deep_ae':
                    # Refer to the deep_ae_summary_results.csv
                    threshold_value = results_array['threshold_value'][i]
                else:
                    threshold_value = 0

                # Test normal data
                TNR = aeu.test_trained_model(test_data=sub_normal_data,
                                             clf_name=model_name,
                                             model_dir_path=trained_model_path,
                                             iteration_number=i,
                                             is_abnormal=False,
                                             threshold_value=threshold_value)

                # Test abnormal data
                TPR = aeu.test_trained_model(test_data=sub_abnormal_data,
                                             clf_name=model_name,
                                             model_dir_path=trained_model_path,
                                             iteration_number=i,
                                             is_abnormal=True,
                                             threshold_value=threshold_value)

                # Compute TP, TN, FP, FN
                #TP = abnormal_ratio
                #TN = normal_ratio
                #FP = 1 - TN
                #FN = 1 - TP

                # Compute TPR and TNR
                #TPR = TP / (TP + FN) = abnormal_ratio
                #TNR = TN / (FP + TN) = normal_ratio

                TPR_list.append(int(TPR*100))
                TNR_list.append(int(TNR*100))

                output_string = '\nTPR = {0:.2f}% and TNR = {1:.2f}%'.format(TPR*100, TNR*100)
                print(output_string)
                print('==============================================================')

            # Get the best one that gives the max value of TPR + TNR
            TPR_list = np.array(TPR_list)
            TNR_list = np.array(TNR_list)
            best_index = np.argmax(TPR_list + TNR_list)
            TPR_best = TPR_list[best_index]
            TNR_best = TNR_list[best_index]
            is_TPR_best = (TPR_best == np.max(TPR_list))
            is_TNR_best = (TNR_best == np.max(TNR_list))

            if is_TPR_best:
                TPR_string = r'\textbf{{{}}}'.format(TPR_best)
            else:
                TPR_string = str(TPR_best)

            if is_TNR_best:
                TNR_string = r'\textbf{{{}}}'.format(TNR_best)
            else:
                TNR_string = str(TNR_best)

            result_file.write(r'& {} & {} '.format(TPR_string, TNR_string))

        result_file.write(r'\\' + '\n')

    # Change the directory back to the initial one
    os.chdir(or_dir_name)
