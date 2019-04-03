"""
Generates a tex file containing the Latex table setup with all the relevant results.
"""
import input_data as data
import rene.analyse_results as rene_ar
import rouen.analyse_results as rouen_ar
import sherbrooke.analyse_results as sherb_ar
import stmarc.analyse_results as stmarc_ar

global_result_path_name = 'global_result/experimental_results_2.tex'

data.make_dir_if_new(global_result_path_name)

# Global result
with open(global_result_path_name, 'w') as global_result_file:
    global_result_file.write(r'\documentclass{article}' + '\n\n')
    
    global_result_file.write(r'\usepackage{multirow}' + '\n')
    
    global_result_file.write(r'\begin{document}' + '\n\n')
    
    global_result_file.write(r'\begin{table}' + '\n')
    global_result_file.write(r'\centering' + '\n')
    global_result_file.write(r'\caption{Results obtained by applying the trained model on the corresponding samples.}' \
                  r'\label{tab2}' + '\n')
    global_result_file.write(r'\begin{tabular}{c|c|c|c||c|c|c|c|c|c|c|c}' + '\n')
    global_result_file.write(r'\multicolumn{4}{c}{} & \multicolumn{8}{c}{Method (\%)} \\ \cline{5-12}' + '\n')
    global_result_file.write(r'\multicolumn{4}{c}{} & \multicolumn{2}{|c|}{OC-SVM} & \multicolumn{2}{c|}{IF} & ' \
                  r'\multicolumn{2}{c|}{AE} & \multicolumn{2}{c}{Deep-AE} \\ \hline' + '\n')
    global_result_file.write(r'Video & Type & Size$_N$ & Size$_A$ & \textnormal{TPR} & \textnormal{TNR} & ' \
                  r'\textnormal{TPR} & \textnormal{TNR} & \textnormal{TPR} & \textnormal{TNR} & ' \
                  r'\textnormal{TPR} & \textnormal{TNR} \\ \hline \hline' + '\n')
    
    sherb_ar.get_comparison_results(global_result_file)
    global_result_file.write(r'\hline' + '\n')
    
    rouen_ar.get_comparison_results(global_result_file)
    global_result_file.write(r'\hline' + '\n')
    
    stmarc_ar.get_comparison_results(global_result_file)
    global_result_file.write(r'\hline' + '\n')
    
    rene_ar.get_comparison_results(global_result_file)
    global_result_file.write(r'\hline' + '\n')
    
    global_result_file.write(r'\end{tabular}' + '\n')
    global_result_file.write(r'\end{table}' + '\n\n')
    
    global_result_file.write(r'\end{document}')
