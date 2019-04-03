"""
This script runs test using the trained models and generates ablation study curves.
"""
import warnings
import ae_utilities as aeu
import numpy as np
import os
import input_data as data
import matplotlib.pyplot as plt


warnings.simplefilter(action='ignore', category=FutureWarning)

abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
normal_name_1 = 'normal_p_data.csv'
normal_file_path_1 = os.path.join(dir_name + '/data', normal_name_1)
abnormal_name_1 = 'abnormal_p_data.csv'
abnormal_file_path_1 = os.path.join(dir_name + '/data', abnormal_name_1)

# Extract normal and abnormal trajectories and export them to array
normal_data_1 = np.genfromtxt(normal_file_path_1, delimiter=',')
abnormal_data_1 = np.genfromtxt(abnormal_file_path_1, delimiter=',')

input_size_1 = len(normal_data_1[0, 1:])

best_layer_type_1 = (128, 64, 32, 16, 8)

# Files containing info of the model and threshold value
model_name_1 = 'new_method_v4_5'
trained_model_path_1 = 'model/' + model_name_1 + '/'
result_path_1 = 'results/' + model_name_1 + '/'

normal_name_2 = 'normal_data.csv'
normal_file_path_2 = os.path.join(dir_name + '/data', normal_name_2)
abnormal_name_2 = 'abnormal_data.csv'
abnormal_file_path_2 = os.path.join(dir_name + '/data', abnormal_name_2)

# Extract normal and abnormal trajectories and export them to array
normal_data_2 = np.genfromtxt(normal_file_path_2, delimiter=',')
abnormal_data_2 = np.genfromtxt(abnormal_file_path_2, delimiter=',')

input_size_2 = len(normal_data_2[0, 1:])

best_layer_type_2 = (256, 128, 64, 32, 16, 8)

# Files containing info of the model and threshold value
model_name_2 = 'new_method_v4_5_3'
trained_model_path_2 = 'model/' + model_name_2 + '/'
result_path_2 = 'results/' + model_name_2 + '/'

global_result_name = 'ablation_study'
result_path = 'results/' + global_result_name + '/'
global_summary_filename = result_path + 'global_ablation_study_2.csv'

data.make_dir_if_new(global_summary_filename)

with open(os.path.join(dir_name, global_summary_filename), 'wb') as summary_file:
    summary_file.write(b'threshold,best_ADA_1,best_NDA_1,best_ae_ADA_1,best_ae_NDA_1,'
                       b'best_ADA_2,best_NDA_2,best_ae_ADA_2,best_ae_NDA_2\n')

best_ADA_list_1 = []
best_NDA_list_1 = []
best_ae_ADA_list_1 = []
best_ae_NDA_list_1 = []

best_ADA_list_2 = []
best_NDA_list_2 = []
best_ae_ADA_list_2 = []
best_ae_NDA_list_2 = []

threshold_values = np.linspace(0.01, 1.0, num=100)

for threshold in threshold_values:
    #======================================================================================================
    print('============================================ No Features ============================================')
    ADA_list_1 = []
    NDA_list_1 = []
    ae_ADA_list_1 = []
    ae_NDA_list_1 = []

    for i in range(aeu.repeat_number):
        print('======================== Iteration {} ========================'.format(i))

        mv4 = aeu.BuildOurMethodV4(original_dim=input_size_1,
                                   hidden_units=best_layer_type_1,
                                   model_dir_path=trained_model_path_1,
                                   iteration_number=i)

        mv4.load_weights()
        # Test the model and the autoencoder separately
        NDA, ae_NDA = mv4.test_complete_model(test_data=normal_data_1, test_ae=True, result_dir_path=result_path_1,
                                              classification_threshold=threshold)
        ADA, ae_ADA = mv4.test_complete_model(test_data=abnormal_data_1, is_abnormal=True, test_ae=True,
                                              result_dir_path=result_path_1, classification_threshold=threshold)

        ADA_list_1.append(int(ADA*100))
        NDA_list_1.append(int(NDA*100))
        ae_ADA_list_1.append(int(ae_ADA*100))
        ae_NDA_list_1.append(int(ae_NDA*100))

        output_string = '\nNDA = {0:.2f}% and ADA = {1:.2f}% (old: ae_NDA = {2:.2f}% and ae_ADA = {3:.2f}%)'\
            .format(NDA*100, ADA*100,ae_NDA*100, ae_ADA*100)
        print(output_string)
        print('==============================================================')

    # Get the best one that gives the max value of TPR + TNR
    NDA_list_1 = np.array(NDA_list_1)
    ADA_list_1 = np.array(ADA_list_1)
    best_index_1 = np.argmax(NDA_list_1 + ADA_list_1)
    NDA_best_1 = NDA_list_1[best_index_1]
    ADA_best_1 = ADA_list_1[best_index_1]

    ae_NDA_list_1 = np.array(ae_NDA_list_1)
    ae_ADA_list_1 = np.array(ae_ADA_list_1)
    ae_best_index_1 = np.argmax(ae_NDA_list_1 + ae_ADA_list_1)
    ae_NDA_best_1 = ae_NDA_list_1[ae_best_index_1]
    ae_ADA_best_1 = ae_ADA_list_1[ae_best_index_1]

    best_ADA_list_1.append(ADA_best_1)
    best_NDA_list_1.append(NDA_best_1)
    best_ae_ADA_list_1.append(ae_ADA_best_1)
    best_ae_NDA_list_1.append(ae_NDA_best_1)

    #======================================================================================================
    print('============================================ With Features ============================================')
    ADA_list_2 = []
    NDA_list_2 = []
    ae_ADA_list_2 = []
    ae_NDA_list_2 = []

    for i in range(aeu.repeat_number):
        print('======================== Iteration {} ========================'.format(i))

        mv4 = aeu.BuildOurMethodV4v2(original_dim=input_size_2,
                                     hidden_units=best_layer_type_2,
                                     model_dir_path=trained_model_path_2,
                                     iteration_number=i)

        mv4.load_weights()
        # Test the model and the autoencoder separately
        NDA, ae_NDA = mv4.test_complete_model(test_data=normal_data_2, test_ae=True, result_dir_path=result_path_2,
                                              classification_threshold=threshold)
        ADA, ae_ADA = mv4.test_complete_model(test_data=abnormal_data_2, is_abnormal=True, test_ae=True,
                                              result_dir_path=result_path_2, classification_threshold=threshold)

        ADA_list_2.append(int(ADA*100))
        NDA_list_2.append(int(NDA*100))
        ae_ADA_list_2.append(int(ae_ADA*100))
        ae_NDA_list_2.append(int(ae_NDA*100))

        output_string = '\nNDA = {0:.2f}% and ADA = {1:.2f}% (old: ae_NDA = {2:.2f}% and ae_ADA = {3:.2f}%)'\
            .format(NDA*100, ADA*100,ae_NDA*100, ae_ADA*100)
        print(output_string)
        print('==============================================================')

    # Get the best one that gives the max value of TPR + TNR
    NDA_list_2 = np.array(NDA_list_2)
    ADA_list_2 = np.array(ADA_list_2)
    best_index_2 = np.argmax(NDA_list_2 + ADA_list_2)
    NDA_best_2 = NDA_list_2[best_index_2]
    ADA_best_2 = ADA_list_2[best_index_2]

    ae_NDA_list_2 = np.array(ae_NDA_list_2)
    ae_ADA_list_2 = np.array(ae_ADA_list_2)
    ae_best_index_2 = np.argmax(ae_NDA_list_2 + ae_ADA_list_2)
    ae_NDA_best_2 = ae_NDA_list_2[ae_best_index_2]
    ae_ADA_best_2 = ae_ADA_list_2[ae_best_index_2]

    best_ADA_list_2.append(ADA_best_2)
    best_NDA_list_2.append(NDA_best_2)
    best_ae_ADA_list_2.append(ae_ADA_best_2)
    best_ae_NDA_list_2.append(ae_NDA_best_2)

    with open(os.path.join(dir_name, global_summary_filename), 'ab') as summary_file:
        np.savetxt(summary_file, np.array([threshold,ADA_best_1,NDA_best_1,ae_ADA_best_1,ae_NDA_best_1,
                                           ADA_best_2,NDA_best_2,ae_ADA_best_2,ae_NDA_best_2]).reshape(1, -1),
                   delimiter=',')

#================================================================================
print('====================================== Generating figures ======================================')
threshold_values = threshold_values*100
plt.rc('text', usetex=True)

fig1 = plt.figure()
plt.plot(threshold_values, best_ae_ADA_list_1, 'k-',
         threshold_values, best_ADA_list_1, 'b--',
         threshold_values, best_ae_ADA_list_2, 'g.-',
         threshold_values, best_ADA_list_2, 'r-')
plt.ylabel('ADA (%)')
plt.xlabel(r'$\tau$ (%)')
plt.legend(['DAE', 'DAE+ALREC', 'DAE+Features', 'DAE+Features+ALREC'], loc='lower left')
figure_name = result_path + global_result_name + '_ADA.pdf'
fig1.savefig(figure_name)

fig2 = plt.figure()
plt.plot(threshold_values, best_ae_NDA_list_1, 'k-',
         threshold_values, best_NDA_list_1, 'b--',
         threshold_values, best_ae_NDA_list_2, 'g.-',
         threshold_values, best_NDA_list_2, 'r-')
plt.ylabel('NDA (%)')
plt.xlabel(r'$\tau$ (%)')
plt.legend(['DAE', 'DAE+ALREC', 'DAE+Features', 'DAE+Features+ALREC'], loc='lower left')
figure_name = result_path + global_result_name + '_NDA.pdf'
fig2.savefig(figure_name)
