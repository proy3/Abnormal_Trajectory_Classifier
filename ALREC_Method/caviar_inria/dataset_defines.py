"""
This script contains some global variables that are used throughout this specific dataset.
"""

# Dataset file names
raw_input_normal_file_path = '/home/travail/datasets/caviar_inria_lab/normal/annotations/'
raw_input_abnormal_file_path = '/home/travail/datasets/caviar_inria_lab/abnormal/annotations/'
normal_data_names = ['br1gt',
                     'br2gt',
                     'br3gt',
                     'br4gt',
                     'bww1gt',
                     'bww2gt',
                     'mc1gt',
                     'ms3ggt',
                     'mws1gt',
                     'mwt1gt',
                     'mwt2gt',
                     'ricgt',
                     'spgt',
                     'wk1gt',
                     'wk2gt',
                     'wk3gt']
abnormal_data_names = ['fcgt',
                       'fomdgt1',
                       'fomdgt2',
                       'fomdgt3',
                       'fra1gt',
                       'fra2gt',
                       'lb1gt',
                       'lb2gt',
                       'lbbcgt',
                       'lbgt',
                       'lbpugt',
                       'rffgt',
                       'rsfgt',
                       'rwgt']
input_raw_image_frame_path = '/home/travail/datasets/caviar_inria_lab/frames/'
input_raw_image_frame_name = 'Walk'
frame_starting_number = 1000
image_frame_extension = 'jpg'

# Used for generating velocities
video_data_fps = 25
