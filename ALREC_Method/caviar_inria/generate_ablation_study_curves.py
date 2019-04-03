"""
Generates with curves with AUC values in the labels
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

global_result_name = 'ablation_study'
result_path = 'results/' + global_result_name + '/'
global_summary_filename = result_path + 'global_ablation_study_2.csv'

# Ref.: https://stackoverflow.com/questions/29451030/why-doesnt-np-genfromtxt-remove-header-while-importing-in-python
with open(global_summary_filename, 'r') as results:
    line = results.readline()
    header = [e for e in line.strip().split(',') if e]

    results_array = np.genfromtxt(results, names=header, dtype=None, delimiter=',')

thresholds = results_array['threshold'][:]
thresholds *= 100

print('====================================== Generating figures ======================================')
plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

#plt.rc('text', usetex=True)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.figsize"] = (8, 3)
plt.rcParams.update({'font.size': 12})

fig1 = plt.figure()
plt.plot(thresholds, results_array['best_ADA_2'][:], 'r-',
         thresholds, results_array['best_ae_ADA_2'][:], 'g.-',
         thresholds, results_array['best_ADA_1'][:], 'b--',
         thresholds, results_array['best_ae_ADA_1'][:], 'k-', linewidth=2.0)
plt.ylabel('ADA')
plt.xlabel('Threshold (%)')
plt.legend(['DAE+Features+ALREC [{}]'.format(int(round(auc(thresholds, results_array['best_ADA_2'][:])))),
            'DAE+Features [{}]'.format(int(round(auc(thresholds, results_array['best_ae_ADA_2'][:])))),
            'DAE+ALREC [{}]'.format(int(round(auc(thresholds, results_array['best_ADA_1'][:])))),
            'DAE [{}]'.format(int(round(auc(thresholds, results_array['best_ae_ADA_1'][:]))))],
           loc='upper right')
figure_name = result_path + global_result_name + '_ADA.pdf'
fig1.savefig(figure_name, bbox_inches='tight', pad_inches=0)

fig2 = plt.figure()
plt.plot(thresholds, results_array['best_ae_NDA_1'][:], 'k-',
         thresholds, results_array['best_NDA_1'][:], 'b--',
         thresholds, results_array['best_ae_NDA_2'][:], 'g.-',
         thresholds, results_array['best_NDA_2'][:], 'r-', linewidth=2.0)
plt.ylabel('NDA')
plt.xlabel('Threshold (%)')
plt.legend(['DAE [{}]'.format(int(round(auc(thresholds, results_array['best_ae_NDA_1'][:])))),
            'DAE+ALREC [{}]'.format(int(round(auc(thresholds, results_array['best_NDA_1'][:])))),
            'DAE+Features [{}]'.format(int(round(auc(thresholds, results_array['best_ae_NDA_2'][:])))),
            'DAE+Features+ALREC [{}]'.format(int(round(auc(thresholds, results_array['best_NDA_2'][:]))))],
           loc='lower right')
figure_name = result_path + global_result_name + '_NDA.pdf'
fig2.savefig(figure_name, bbox_inches='tight', pad_inches=0)

print('====================================== Finished ======================================')
