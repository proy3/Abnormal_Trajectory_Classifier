"""
Generate the background image with trajectories including normal and abnormal ones.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import input_data as data
import abnormal_data_generation as adg
import dataset_defines as dd

# Extract trajectories and export data to array
dataset_file = data.extract_augment_and_export_data(raw_input_file_all=dd.raw_input_file_all_2,
                                                    input_raw_image_frame_path=dd.input_raw_image_frame_path_2,
                                                    raw_input_file_names=dd.raw_input_file_names_2,
                                                    video_data_fps=dd.video_data_fps,
                                                    generate_graph=True,
                                                    show_graph=True)

# Generate abnormal data
abnormal_data = adg.generate_abnormal_data_from_raw_data(overwrite_data=True,
                                                         generate_graph=True, show_graph=True)