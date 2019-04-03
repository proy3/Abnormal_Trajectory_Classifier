# ----------------------------------------------------
# Datasets processing by extracting sequences of tracklets.
# Created by: Pankaj Raj Roy
# Last Modified: Friday, June 01, 2018
# LITIV - Polytechnique Montreal
# ==============================================================================

import sqlite3
import pandas.io.sql as sql
import numpy as np
import os
from scipy.ndimage import zoom
import random
import trajectory_viewer as tv
import time


raw_table_name = 'bounding_boxes'

sub_trajectory_size = 31

# Root directory:
dir_name = os.getcwd()

# Random seed number for reproducing the sequence of randomness
random_seed_number = 99999

# Number of augmented trajectory data per trajectory
number_of_augmented_data_per_raw_data = 50

# Slide the trajectory frame window by this value
trajectory_stride = 10

# Augmented object id code
# Ex.: 901
#      9: Augmented object id code
#      0: Raw object id
#      1: Second generated augmented data of the raw data
augmented_object_id_code = 9

train_sample_percentage = 80


def split_dataset_uniformly(dataset):
    random_shuffle_seed = int((time.time() - int(time.time()))*1e6)
    np.random.seed(random_shuffle_seed)

    n_object_label = 3
    is_first = True
    train_data = []
    valid_data = []
    for i in range(n_object_label):
        sub_dataset = dataset[dataset[:,0] == i]

        if len(sub_dataset) == 0:
            continue

        np.random.shuffle(sub_dataset)

        # Get the index that separates from training data to validation data
        train_sep_index = int((train_sample_percentage / 100.0) * sub_dataset.shape[0])

        sub_train_data = sub_dataset[:train_sep_index,:]
        sub_validation_data = sub_dataset[train_sep_index:,:]

        if is_first:
            train_data = sub_train_data
            valid_data = sub_validation_data
            is_first = False
        else:
            train_data = np.concatenate((train_data, sub_train_data), axis=0)
            valid_data = np.concatenate((valid_data, sub_validation_data), axis=0)

    np.random.shuffle(train_data)

    return train_data, valid_data, random_shuffle_seed


def make_dir_if_new(filename):
    directory = os.path.join(dir_name, filename[:filename.rfind('/')])
    if not os.path.exists(directory):
        os.makedirs(directory)


def _rearrange_data(object_id, object_label, x, y, v_x, v_y, generate_graph, trajectory_image):
    """
    Generates an array formed by the input data in the following format:
    [[Object_id, x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1, ..., x_15, y_15, v_x_15, v_y_15]
     [Object_id, x_16, y_16, v_x_16, v_y_16, x_17, y_17, v_x_17, v_y_17, ..., x_31, y_31, v_x_31, v_y_31]
     ...]
    :param object_id:
    :param x:
    :param y:
    :param v_x:
    :param v_y:
    :return:
    """
    expected_input_size = sub_trajectory_size + int(np.ceil(float(x.shape[0] - sub_trajectory_size) /
                                                            trajectory_stride)) * trajectory_stride

    rows_number = (expected_input_size - sub_trajectory_size) / trajectory_stride + 1

    if expected_input_size > x.shape[0]:
        # Resize all the input array to the expected size
        zoom_ratio = expected_input_size / float(x.shape[0])
        x = zoom(x, (zoom_ratio, 1))
        y = zoom(y, (zoom_ratio, 1))
        v_x = zoom(v_x, (zoom_ratio, 1))
        v_y = zoom(v_y, (zoom_ratio, 1))

    array_data = np.full((rows_number, 1), object_id)
    array_data = np.hstack((array_data, np.full((rows_number, 1), object_label)))

    if generate_graph:
        for i in range(x.shape[0] / sub_trajectory_size):
            trajectory_image.add_trajectory(x[i*sub_trajectory_size:(i+1)*sub_trajectory_size].flatten(),
                                            y[i*sub_trajectory_size:(i+1)*sub_trajectory_size].flatten(),
                                            line_width=object_label+1,
                                            line_color=['red', 'limegreen', 'c'][object_label])

    for i in range(sub_trajectory_size):
        array_data = np.hstack((array_data,
                                x[i::trajectory_stride][:rows_number,:],
                                y[i::trajectory_stride][:rows_number,:],
                                v_x[i::trajectory_stride][:rows_number,:],
                                v_y[i::trajectory_stride][:rows_number,:]))

    return array_data


def extract_and_put_transformed_data(raw_input_file_all,
                                     input_raw_image_frame_path,
                                     raw_input_file_names,
                                     video_data_fps,
                                     changed_trajectories,
                                     generate_graph=False, show_graph=False):
    """
    Extracts the trajectories positions and the related velocities and
    exports those to two CSV files, one for training and the other for
    validation of NN.
    Note that the training_sample_percentage of total sample will be used for training
    and (100 - training_sample_percentage) will be used for validation.
    :return: names of the exported CSV files.
    """
    # Print a starting message
    print("====================================================")
    print("Exporting modified trajectories from raw dataset and")
    print("Exporting them to CSV files.")

    # Trajectory image viewer
    trajectory_image = tv.ImageViewer(input_raw_image_frame_path)

    # Format of the data: [Object_id, Object_label x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1, x_2, y_2, ...]

    data_filename = '_real_abnormal_2'

    # Extract trajectories and export to CSV file
    dataset_file_name = 'data/'+raw_input_file_all[raw_input_file_all.rfind('/')+1:raw_input_file_all.rfind('.')]\
                        + data_filename + '.csv'

    directory = os.path.join(dir_name, dataset_file_name[:dataset_file_name.rfind('/')])
    if not os.path.exists(directory):
        os.makedirs(directory)

    object_label = 0

    first_call = True

    trajectory_index = 0

    for raw_input_file in raw_input_file_names:
        # Skip if the string is empty
        if not raw_input_file:
            object_label += 1
            continue

        # Convert sqlite dataset to dictionary
        conn = sqlite3.connect(raw_input_file)
        raw_dataset = sql.read_sql('select * from {tn}'.format(tn=raw_table_name), conn)
        conn.close()

        # Maximum number of objects
        n_objects = int((raw_dataset.loc[raw_dataset['object_id'].idxmax()])['object_id']) + 1

        for object_id in range(0, n_objects):
            # Extract rows for the particular object_id
            df = raw_dataset.loc[raw_dataset['object_id'] == object_id]

            # Extract frame numbers
            frame_number = df['frame_number'].values

            x_centered = changed_trajectories[trajectory_index].trajectory.get_xdata()
            y_centered = changed_trajectories[trajectory_index].trajectory.get_ydata()
            total_scale_factor = changed_trajectories[trajectory_index].total_scale_factor

            x_1 = x_centered[:-1]
            x_2 = x_centered[1:]
            y_1 = y_centered[:-1]
            y_2 = y_centered[1:]
            f_1 = frame_number[:-1]
            f_2 = frame_number[1:]

            v_x = video_data_fps * (x_2 - x_1) / (total_scale_factor * (f_2 - f_1))
            v_y = video_data_fps * (y_2 - y_1) / (total_scale_factor * (f_2 - f_1))

            # Concatenate data
            x_1 = np.array(x_1).reshape(-1, 1)
            y_1 = np.array(y_1).reshape(-1, 1)
            trajectory_index += 1

            v_x = np.array(v_x).reshape(-1, 1)
            v_y = np.array(v_y).reshape(-1, 1)

            array_dataset = _rearrange_data(object_id, object_label, x_1, y_1, v_x, v_y,
                                            generate_graph=generate_graph,
                                            trajectory_image=trajectory_image)

            open_file_mode = 'wb' if first_call else 'ab'

            first_call = False

            with open(os.path.join(dir_name, dataset_file_name), open_file_mode) as datafile:
                np.savetxt(datafile, array_dataset, fmt="%.2f", delimiter=",")

        object_label += 1

    # Save and Show the trajectory image
    if generate_graph:
        trajectory_name = '_real_abnormal_trajectories'
        trajectory_image_name = directory + '/' + \
                                raw_input_file_all[raw_input_file_all.rfind('/')+1:raw_input_file_all.rfind('.')] + \
                                trajectory_name + '.pdf'
        trajectory_image.save_image(os.path.join(dir_name, trajectory_image_name))
    if generate_graph and show_graph:
        trajectory_image.show_image()

    # Print finishing message
    print("                                          ---> Done!")
    print("====================================================")

def extract_augment_and_export_data(raw_input_file_all,
                                    input_raw_image_frame_path,
                                    raw_input_file_names,
                                    video_data_fps,
                                    generate_graph = False, show_graph = False):
    """
    Generate augmented data from the dataset.
    :return: Name of the augmented dataset
    """
    # Print a starting message
    print("=========================================================")
    print("Extracting trajectory data from raw dataset,")
    print("Generating augmented trajectory data and")
    print("Exporting them to CSV file.")

    # Trajectory image viewer
    trajectory_image = tv.ImageViewer(input_raw_image_frame_path)

    # Format of the data: [Object_id, Object_label x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1, x_2, y_2, ...]

    # Extract trajectories and export to CSV file
    dataset_file_name = 'data/'+raw_input_file_all[raw_input_file_all.rfind('/')+1:raw_input_file_all.rfind('.')]\
                        +'_data.csv'

    directory = os.path.join(dir_name, dataset_file_name[:dataset_file_name.rfind('/')])
    if not os.path.exists(directory):
        os.makedirs(directory)

    object_label = 0

    # Initialize seed for random rand in order to have the same sequence of randomness every time this is called
    random.seed(random_seed_number)

    first_call = True

    for raw_input_file in raw_input_file_names:
        # Skip if the string is empty
        if not raw_input_file:
            object_label += 1
            continue

        # Convert sqlite dataset to dictionary
        conn = sqlite3.connect(raw_input_file)
        raw_dataset = sql.read_sql('select * from {tn}'.format(tn=raw_table_name), conn)
        conn.close()

        # Maximum number of objects
        n_objects = int((raw_dataset.loc[raw_dataset['object_id'].idxmax()])['object_id']) + 1

        for object_id in range(0, n_objects):
            # Extract rows for the particular object_id
            df = raw_dataset.loc[raw_dataset['object_id'] == object_id]

            # Extract trajectory
            x_top_left = df['x_top_left'].values
            y_top_left = df['y_top_left'].values
            x_bottom_right = df['x_bottom_right'].values
            y_bottom_right = df['y_bottom_right'].values
            frame_number = df['frame_number'].values

            x_centered = (x_top_left + x_bottom_right) / 2
            y_centered = (y_top_left + y_bottom_right) / 2

            x_1_r = x_centered[:-1]
            x_2_r = x_centered[1:]
            y_1_r = y_centered[:-1]
            y_2_r = y_centered[1:]
            f_1 = frame_number[:-1]
            f_2 = frame_number[1:]

            v_x = video_data_fps * (x_2_r - x_1_r) / (f_2 - f_1)
            v_y = video_data_fps * (y_2_r - y_1_r) / (f_2 - f_1)

            # Concatenate data
            x_1 = np.array(x_1_r).reshape(-1, 1)
            y_1 = np.array(y_1_r).reshape(-1, 1)
            v_x = np.array(v_x).reshape(-1, 1)
            v_y = np.array(v_y).reshape(-1, 1)

            array_dataset = _rearrange_data(object_id, object_label, x_1, y_1, v_x, v_y,
                                            generate_graph=generate_graph,
                                            trajectory_image=trajectory_image)

            open_file_mode = 'wb' if first_call else 'ab'

            first_call = False

            with open(os.path.join(dir_name, dataset_file_name), open_file_mode) as datafile:
                np.savetxt(datafile, array_dataset, fmt="%.2f", delimiter=",")

            # Generate augmented positions
            for i in range(number_of_augmented_data_per_raw_data):
                x_a = np.array([random.randint(min(x[0],x[1]), max(x[0],x[1]))
                                for x in np.array([x_1_r,x_2_r]).astype(int).T])
                y_a = np.array([random.randint(min(y[0],y[1]), max(y[0],y[1]))
                                for y in np.array([y_1_r,y_2_r]).astype(int).T])

                x_a_1 = x_a[:-1]
                x_a_2 = x_a[1:]
                y_a_1 = y_a[:-1]
                y_a_2 = y_a[1:]

                v_x_a = video_data_fps * (x_a_2 - x_a_1) / (f_2[:-1] - f_1[:-1])
                v_y_a = video_data_fps * (y_a_2 - y_a_1) / (f_2[:-1] - f_1[:-1])

                # Concatenate data
                x_a_1 = np.array(x_a_1).reshape(-1, 1)
                y_a_1 = np.array(y_a_1).reshape(-1, 1)
                v_x_a = np.array(v_x_a).reshape(-1, 1)
                v_y_a = np.array(v_y_a).reshape(-1, 1)

                augmented_object_id = int(str(augmented_object_id_code) + str(object_id) + str(i))

                array_dataset = _rearrange_data(augmented_object_id, object_label, x_a_1, y_a_1, v_x_a, v_y_a,
                                                generate_graph=False,
                                                trajectory_image=trajectory_image)

                with open(os.path.join(dir_name, dataset_file_name), 'ab') as datafile:
                    np.savetxt(datafile, array_dataset, fmt="%.2f", delimiter=",")

        object_label += 1

    # Save and Show the trajectory image
    if generate_graph:
        trajectory_image_name = directory + '/' + \
                                raw_input_file_all[raw_input_file_all.rfind('/')+1:raw_input_file_all.rfind('.')] + \
                                '_normal_trajectories.pdf'
        trajectory_image.save_image(os.path.join(dir_name, trajectory_image_name))
    if generate_graph and show_graph:
        trajectory_image.show_image()

    # Print finishing message
    print("                                               ---> Done!")
    print("=========================================================")

    return dataset_file_name

#Test
#extract_augment_and_export_data(generate_graph=True,show_graph=True)
