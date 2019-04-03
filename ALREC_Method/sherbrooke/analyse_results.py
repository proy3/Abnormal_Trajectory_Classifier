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

    list_of_model_names = ['one_class_svm','isolation_forest','single_ae','deep_ae','new_method_v4_5']

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

    row_string = r'\multirow{{{}}}{{*}}{{St-Marc}}'.format(n_labels)

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

        NDA_models = []
        ADA_models = []

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

            ADA_list = []
            NDA_list = []

            for i in range(aeu.repeat_number):
                print('======================== Iteration {} ========================'.format(i))

                if model_name == 'single_ae' or model_name == 'deep_ae':
                    # Refer to the deep_ae_summary_results.csv
                    threshold_value = results_array['threshold_value'][i]
                else:
                    threshold_value = 0

                if model_name == 'new_method_v4_5':
                    mv4 = aeu.BuildOurMethodV4(original_dim=dataset.shape[1],
                                               model_dir_path=trained_model_path,
                                               iteration_number=i)

                    mv4.load_weights()
                    # Test the model and the autoencoder separately
                    _, NDA = mv4.test_model(test_data=sub_normal_data)
                    _, ADA = mv4.test_model(test_data=sub_abnormal_data, is_abnormal=True)
                else:
                    # Test normal data
                    NDA = aeu.test_trained_model(test_data=sub_normal_data,
                                                 clf_name=model_name,
                                                 model_dir_path=trained_model_path,
                                                 iteration_number=i,
                                                 is_abnormal=False,
                                                 threshold_value=threshold_value)

                    # Test abnormal data
                    ADA = aeu.test_trained_model(test_data=sub_abnormal_data,
                                                 clf_name=model_name,
                                                 model_dir_path=trained_model_path,
                                                 iteration_number=i,
                                                 is_abnormal=True,
                                                 threshold_value=threshold_value)

                ADA_list.append(int(ADA*100))
                NDA_list.append(int(NDA*100))

                output_string = '\nNDA = {0:.2f}% and ADA = {1:.2f}%'.format(NDA*100, ADA*100)
                print(output_string)
                print('==============================================================')

            # Get the best one that gives the max value of TPR + TNR
            NDA_list = np.array(NDA_list)
            ADA_list = np.array(ADA_list)
            best_index = np.argmax(NDA_list + ADA_list)
            NDA_best = NDA_list[best_index]
            ADA_best = ADA_list[best_index]

            NDA_models.append(NDA_best)
            ADA_models.append(ADA_best)

        NDA_models = np.array(NDA_models)
        ADA_models = np.array(ADA_models)

        for i in range(NDA_models):
            is_NDA_best = (NDA_models[i] == np.max(NDA_models))
            is_ADA_best = (ADA_models[i] == np.max(ADA_models))

            if is_NDA_best:
                NDA_string = r'\textbf{{{}}}'.format(NDA_models[i])
            else:
                NDA_string = str(NDA_models[i])

            if is_ADA_best:
                ADA_string = r'\textbf{{{}}}'.format(ADA_models[i])
            else:
                ADA_string = str(ADA_models[i])

            result_file.write(r'& {} & {} '.format(NDA_string, ADA_string))

        result_file.write(r'\\' + '\n')

    # Change the directory back to the initial one
    os.chdir(or_dir_name)
