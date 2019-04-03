"""
This script contains some global variables that are used throughout this specific dataset.
"""

# Dataset file names
raw_input_file_all = '/home/travail/datasets/urban_tracker/sherbrooke/sherbrooke_annotations/sherbrooke_gt.sqlite'
input_raw_image_frame_path = '/home/travail/datasets/urban_tracker/sherbrooke/sherbrooke_frames/'
raw_input_file_peds = raw_input_file_all[:raw_input_file_all.rfind('.')] + '_pedestrians.sqlite'
raw_input_file_cars = raw_input_file_all[:raw_input_file_all.rfind('.')] + '_cars.sqlite'
raw_input_file_names = [raw_input_file_peds, raw_input_file_cars]

# Used for generating velocities
video_data_fps = 30

best_deep_ae_model = 58
