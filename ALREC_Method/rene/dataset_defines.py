"""
This script contains some global variables that are used throughout this specific dataset.
"""

# Dataset file names
raw_input_file_all = '/home/travail/datasets/urban_tracker/rene/rene_annotations/rene_gt.sqlite'
input_raw_image_frame_path = '/home/travail/datasets/urban_tracker/rene/rene_frames/'
raw_input_file_peds = ''
raw_input_file_cars = raw_input_file_all[:raw_input_file_all.rfind('.')] + '_cars.sqlite'
raw_input_file_bike = raw_input_file_all[:raw_input_file_all.rfind('.')] + '_bike.sqlite'
raw_input_file_names = [raw_input_file_peds, raw_input_file_cars, raw_input_file_bike]

raw_input_file_all_2 = 'C:/Users/panka/PyCharmProjects/Dataset/rene_annotations/rene_annotations/rene_gt.sqlite'
input_raw_image_frame_path_2 = 'C:/Users/panka/PyCharmProjects/Dataset/rene_frames/rene_frames/'
raw_input_file_cars_2 = raw_input_file_all_2[:raw_input_file_all_2.rfind('.')] + '_cars.sqlite'
raw_input_file_bike_2 = raw_input_file_all_2[:raw_input_file_all_2.rfind('.')] + '_bike.sqlite'
raw_input_file_names_2 = [raw_input_file_peds, raw_input_file_cars_2, raw_input_file_bike_2]

# Used for generating velocities
video_data_fps = 30

best_deep_ae_model = 28
