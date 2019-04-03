"""
Tests the new ALREC (Adversarially Learned Reconstruction Error Classifier) on complete trajectories (a hole object's
trajectory is considered as a complete trajectory.
"""
import warnings
import ae_utilities as aeu
import numpy as np
import os


warnings.simplefilter(action='ignore', category=FutureWarning)

abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
normal_name = 'normal_p_data.csv'
normal_file_path = os.path.join(dir_name + '/data', normal_name)
abnormal_name = 'abnormal_p_data.csv'
abnormal_file_path = os.path.join(dir_name + '/data', abnormal_name)

# Extract normal and abnormal trajectories and export them to array
normal_data = np.genfromtxt(normal_file_path, delimiter=',')
abnormal_data = np.genfromtxt(abnormal_file_path, delimiter=',')

input_size = len(normal_data[0, 1:])

best_layer_type = (128, 64, 32, 16, 8)

# Files containing info of the model and threshold value
model_name = 'new_method_v4_5'
trained_model_path = 'model/' + model_name + '/'
result_path = 'results/' + model_name + '/'

global_summary_filename = result_path + 'global_complete_summary.log'

ADA_list = []
NDA_list = []
ae_ADA_list = []
ae_NDA_list = []

repeat_number = 20

for i in range(repeat_number):
    print('======================== Iteration {} ========================'.format(i))

    mv4 = aeu.BuildOurMethodV4(original_dim=input_size,
                               hidden_units=best_layer_type,
                               model_dir_path=trained_model_path,
                               iteration_number=i)

    mv4.load_weights()
    # Test the model and the autoencoder separately
    NDA, ae_NDA = mv4.test_complete_model(test_data=normal_data, test_ae=True, result_dir_path=result_path)
    ADA, ae_ADA = mv4.test_complete_model(test_data=abnormal_data, is_abnormal=True, test_ae=True,
                                          result_dir_path=result_path)

    # Test false positives
    fp_NDA, fp_ae_NDA = mv4.test_complete_model(test_data=abnormal_data, test_ae=True, result_dir_path=result_path)
    fp_ADA, fp_ae_ADA = mv4.test_complete_model(test_data=normal_data, is_abnormal=True, test_ae=True,
                                                result_dir_path=result_path)

    ADA_list.append(int(ADA*100))
    NDA_list.append(int(NDA*100))
    ae_ADA_list.append(int(ae_ADA*100))
    ae_NDA_list.append(int(ae_NDA*100))

    output_string = '\nNDA = {0:.2f}% and ADA = {1:.2f}% (old: ae_NDA = {2:.2f}% and ae_ADA = {3:.2f}%)'\
        .format(NDA*100, ADA*100,ae_NDA*100, ae_ADA*100)
    output_string += '\nfp_NDA = {0:.2f}% and fp_ADA = {1:.2f}% (old: fp_ae_NDA = {2:.2f}% and fp_ae_ADA = {3:.2f}%)'\
        .format(fp_NDA*100, fp_ADA*100,fp_ae_NDA*100, fp_ae_ADA*100)
    print(output_string)
    print('==============================================================')

# Get the best one that gives the max value of TPR + TNR
NDA_list = np.array(NDA_list)
ADA_list = np.array(ADA_list)
best_index = np.argmax(NDA_list + ADA_list)
NDA_best = NDA_list[best_index]
ADA_best = ADA_list[best_index]

ae_NDA_list = np.array(ae_NDA_list)
ae_ADA_list = np.array(ae_ADA_list)
ae_best_index = np.argmax(ae_NDA_list + ae_ADA_list)
ae_NDA_best = ae_NDA_list[ae_best_index]
ae_ADA_best = ae_ADA_list[ae_best_index]

# Global summary
global_summary_file = open(global_summary_filename, 'w')

output_string = 'Global complete summary of abnormal trajectory detection with our new method v.4\n'
output_string += '--------------------------------------------------------------------------------\n'
output_string += 'Taking the best one with model_{}, using layer type {},\n'.format(best_index, best_layer_type)
output_string += '\t{:.2f}% of normal trajectory samples are detected as normal;\n'.format(
    NDA_best)
output_string += '\t{:.2f}% of abnormal trajectory samples are detected as abnormal.\n'.format(
    ADA_best)
output_string += '--------------------------------------------------------------------------------\n'
output_string += 'On average, using layer type {},\n'.format(best_layer_type)
output_string += '\t{:.2f}% of normal trajectory samples are detected as normal;\n'.format(
    np.mean(NDA_list))
output_string += '\t{:.2f}% of abnormal trajectory samples are detected as abnormal.\n'.format(
    np.mean(ADA_list))
output_string += '-----------------------------------------------------------------------\n'
output_string += 'On maximum, using layer type {},\n'.format(best_layer_type)
output_string += '\t{:.2f}% of normal trajectory samples are detected as normal;\n'.format(
    np.max(NDA_list))
output_string += '\t{:.2f}% of abnormal trajectory samples are detected as abnormal.\n'.format(
    np.max(ADA_list))
output_string += '-----------------------------------------------------------------------\n'
output_string += 'On minimum, using layer type {},\n'.format(best_layer_type)
output_string += '\t{:.2f}% of normal trajectory samples are detected as normal;\n'.format(
    np.min(NDA_list))
output_string += '\t{:.2f}% of abnormal trajectory samples are detected as abnormal.\n'.format(
    np.min(ADA_list))
output_string += '-----------------------------------------------------------------------\n'

output_string += 'Global complete summary of abnormal trajectory detection with DAE only\n'
output_string += '--------------------------------------------------------------------------------\n'
output_string += 'Taking the best one with model_{}, using layer type {},\n'.format(ae_best_index, best_layer_type)
output_string += '\t{:.2f}% of normal trajectory samples are detected as normal;\n'.format(
    ae_NDA_best)
output_string += '\t{:.2f}% of abnormal trajectory samples are detected as abnormal.\n'.format(
    ae_ADA_best)
output_string += '--------------------------------------------------------------------------------\n'
output_string += 'On average, using layer type {},\n'.format(best_layer_type)
output_string += '\t{:.2f}% of normal trajectory samples are detected as normal;\n'.format(
    np.mean(ae_NDA_list))
output_string += '\t{:.2f}% of abnormal trajectory samples are detected as abnormal.\n'.format(
    np.mean(ae_ADA_list))
output_string += '-----------------------------------------------------------------------\n'
output_string += 'On maximum, using layer type {},\n'.format(best_layer_type)
output_string += '\t{:.2f}% of normal trajectory samples are detected as normal;\n'.format(
    np.max(ae_NDA_list))
output_string += '\t{:.2f}% of abnormal trajectory samples are detected as abnormal.\n'.format(
    np.max(ae_ADA_list))
output_string += '-----------------------------------------------------------------------\n'
output_string += 'On minimum, using layer type {},\n'.format(best_layer_type)
output_string += '\t{:.2f}% of normal trajectory samples are detected as normal;\n'.format(
    np.min(ae_NDA_list))
output_string += '\t{:.2f}% of abnormal trajectory samples are detected as abnormal.\n'.format(
    np.min(ae_ADA_list))
output_string += '-----------------------------------------------------------------------\n'

global_summary_file.write(output_string)
print(output_string)

global_summary_file.close()
