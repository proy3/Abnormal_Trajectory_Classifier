"""
Extracts the CAVIAR INRIA laboratoire dataset from XML format to numpy array format.
"""
import numpy as np
import os
import random
import trajectory_viewer as tv
import xml.etree.ElementTree as ET
import input_data as data
from caviar_inria import dataset_defines as dd


def extract_augment_and_export_positions_data(is_normal=True,
                                              generate_graph=False, show_graph=False):
    """
        Generate augmented data from the dataset.
        :return: Name of the augmented dataset
        """
    data_type_name = 'normal' if is_normal else 'abnormal'

    # Print a starting message
    print("=========================================================")
    print("Extracting {} trajectory positions data from raw dataset,".format(data_type_name))
    print("Generating {} augmented trajectory data and".format(data_type_name))
    print("Exporting them to CSV file.")

    # Trajectory image viewer
    trajectory_image = tv.ImageViewer(dd.input_raw_image_frame_path, dd.input_raw_image_frame_name,
                                      dd.frame_starting_number, is_caviar_data=True)

    # Format of the data: [Object_id, Object_label, x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1, x_2, y_2, ...]

    # Extract trajectories and export to CSV file
    data_file_name = 'data/' + data_type_name + '_p_data.csv'

    if is_normal:
        gt_names = dd.normal_data_names
        gt_pathname = dd.raw_input_normal_file_path
    else:
        gt_names = dd.abnormal_data_names
        gt_pathname = dd.raw_input_abnormal_file_path

    directory = os.path.join(data.dir_name, data_file_name[:data_file_name.rfind('/')])
    if not os.path.exists(directory):
        os.makedirs(directory)

    object_label = 0

    # Initialize seed for random rand in order to have the same sequence of randomness every time this is called
    random.seed(data.random_seed_number)

    first_call = True

    for gt_filename in gt_names:
        xml_filename = gt_pathname + gt_filename + '.xml'
        tree = ET.parse(xml_filename)
        root = tree.getroot()

        data_bb_by_frame = []
        for frame in root.findall('frame'):
            object = frame.find('objectlist').find('object')
            if object is None:
                continue
            box = object.find('box')
            if box is None:
                continue
            data_bb = dict(fn=int(frame.get('number')))
            data_bb.update(id=int(object.get('id')))
            data_bb.update(xc=int(box.get('xc')))
            data_bb.update(yc=int(box.get('yc')))
            data_bb_by_frame.append(data_bb)

        max_id = max([item['id'] for item in data_bb_by_frame])
        for object_id in range(0, max_id+1):
            o_data = list(filter(lambda item: item['id'] == object_id, data_bb_by_frame))

            if not o_data:
                continue

            x_centered = np.array([item['xc'] for item in o_data])
            y_centered = np.array([item['yc'] for item in o_data])
            frame_number = np.array([item['fn'] for item in o_data])

            x_1_r = x_centered[:-1]
            x_2_r = x_centered[1:]
            y_1_r = y_centered[:-1]
            y_2_r = y_centered[1:]
            f_1 = frame_number[:-1]
            f_2 = frame_number[1:]

            v_x = dd.video_data_fps * (x_2_r - x_1_r) / (f_2 - f_1)
            v_y = dd.video_data_fps * (y_2_r - y_1_r) / (f_2 - f_1)

            # Concatenate data
            x_1 = np.array(x_1_r).reshape(-1, 1)
            y_1 = np.array(y_1_r).reshape(-1, 1)
            v_x = np.array(v_x).reshape(-1, 1)
            v_y = np.array(v_y).reshape(-1, 1)

            array_data = data._rearrange_data(object_id, object_label, x_1, y_1, v_x, v_y,
                                              generate_graph=generate_graph,
                                              trajectory_image=trajectory_image)

            open_file_mode = 'wb' if first_call else 'ab'

            first_call = False

            with open(os.path.join(data.dir_name, data_file_name), open_file_mode) as datafile:
                np.savetxt(datafile, array_data, fmt="%.2f", delimiter=",")

            # Generate augmented positions
            for i in range(data.number_of_augmented_data_per_raw_data):
                x_a = np.array([random.randint(min(x[0], x[1]), max(x[0], x[1]))
                                for x in np.array([x_1_r, x_2_r]).astype(int).T])
                y_a = np.array([random.randint(min(y[0], y[1]), max(y[0], y[1]))
                                for y in np.array([y_1_r, y_2_r]).astype(int).T])

                x_a_1 = x_a[:-1]
                x_a_2 = x_a[1:]
                y_a_1 = y_a[:-1]
                y_a_2 = y_a[1:]

                v_x_a = dd.video_data_fps * (x_a_2 - x_a_1) / (f_2[:-1] - f_1[:-1])
                v_y_a = dd.video_data_fps * (y_a_2 - y_a_1) / (f_2[:-1] - f_1[:-1])

                # Concatenate data
                x_a_1 = np.array(x_a_1).reshape(-1, 1)
                y_a_1 = np.array(y_a_1).reshape(-1, 1)
                v_x_a = np.array(v_x_a).reshape(-1, 1)
                v_y_a = np.array(v_y_a).reshape(-1, 1)

                #augmented_object_id = int(str(data.augmented_object_id_code) + str(object_id) + str(i))
                augmented_object_id = object_id

                array_data = data._rearrange_data(augmented_object_id, object_label, x_a_1, y_a_1, v_x_a, v_y_a,
                                                  generate_graph=False,
                                                  trajectory_image=trajectory_image)

                with open(os.path.join(data.dir_name, data_file_name), 'ab') as datafile:
                    np.savetxt(datafile, array_data, fmt="%.2f", delimiter=",")

    # Save and Show the trajectory image
    if generate_graph:
        trajectory_image_name = directory + '/' + data_type_name + '_trajectories.pdf'
        trajectory_image.save_image(os.path.join(data.dir_name, trajectory_image_name))
    if generate_graph and show_graph:
        trajectory_image.show_image()

    # Print finishing message
    print("                                               ---> Done!")
    print("=========================================================")

    return data_file_name


def extract_augment_and_export_data(is_normal=True,
                                    generate_graph=False, show_graph=False):
    """
        Generate augmented data from the dataset.
        :return: Name of the augmented dataset
        """
    data_type_name = 'normal' if is_normal else 'abnormal'

    # Print a starting message
    print("=========================================================")
    print("Extracting {} trajectory features data from raw dataset,".format(data_type_name))
    print("Generating {} augmented trajectory data and".format(data_type_name))
    print("Exporting them to CSV file.")

    # Trajectory image viewer
    trajectory_image = tv.ImageViewer(dd.input_raw_image_frame_path, dd.input_raw_image_frame_name,
                                      dd.frame_starting_number, is_caviar_data=True)

    # Format of the data: [Object_id, Object_label, x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1, x_2, y_2, ...]

    # Extract trajectories and export to CSV file
    data_file_name = 'data/' + data_type_name + '_data.csv'

    if is_normal:
        gt_names = dd.normal_data_names
        gt_pathname = dd.raw_input_normal_file_path
    else:
        gt_names = dd.abnormal_data_names
        gt_pathname = dd.raw_input_abnormal_file_path

    directory = os.path.join(data.dir_name, data_file_name[:data_file_name.rfind('/')])
    if not os.path.exists(directory):
        os.makedirs(directory)

    object_label = 0

    # Initialize seed for random rand in order to have the same sequence of randomness every time this is called
    random.seed(data.random_seed_number)

    first_call = True

    for gt_filename in gt_names:
        xml_filename = gt_pathname + gt_filename + '.xml'
        tree = ET.parse(xml_filename)
        root = tree.getroot()

        data_bb_by_frame = []
        for frame in root.findall('frame'):
            object = frame.find('objectlist').find('object')
            if object is None:
                continue
            box = object.find('box')
            if box is None:
                continue
            data_bb = dict(fn=int(frame.get('number')))
            data_bb.update(id=int(object.get('id')))
            data_bb.update(o=int(object.find('orientation').text))
            data_bb.update(h=int(box.get('h')))
            data_bb.update(w=int(box.get('w')))
            data_bb.update(xc=int(box.get('xc')))
            data_bb.update(yc=int(box.get('yc')))
            data_bb_by_frame.append(data_bb)

        max_id = max([item['id'] for item in data_bb_by_frame])
        for object_id in range(0, max_id+1):
            o_data = list(filter(lambda item: item['id'] == object_id, data_bb_by_frame))

            if not o_data:
                continue

            x_centered = np.array([item['xc'] for item in o_data])
            y_centered = np.array([item['yc'] for item in o_data])
            frame_number = np.array([item['fn'] for item in o_data])
            bb_h = np.array([item['h'] for item in o_data])
            bb_w = np.array([item['w'] for item in o_data])
            theta = np.array([item['o'] for item in o_data])

            x_1_r = x_centered[:-1]
            x_2_r = x_centered[1:]
            y_1_r = y_centered[:-1]
            y_2_r = y_centered[1:]
            f_1 = frame_number[:-1]
            f_2 = frame_number[1:]
            h_1_r = bb_h[:-1]
            h_2_r = bb_h[1:]
            w_1_r = bb_w[:-1]
            w_2_r = bb_w[1:]
            o_1_r = theta[:-1]
            o_2_r = theta[1:]

            v_x = dd.video_data_fps * (x_2_r - x_1_r) / (f_2 - f_1)
            v_y = dd.video_data_fps * (y_2_r - y_1_r) / (f_2 - f_1)

            # Concatenate data
            x_1 = np.array(x_1_r).reshape(-1, 1)
            y_1 = np.array(y_1_r).reshape(-1, 1)
            v_x = np.array(v_x).reshape(-1, 1)
            v_y = np.array(v_y).reshape(-1, 1)
            h_1 = np.array(h_1_r).reshape(-1, 1)
            w_1 = np.array(w_1_r).reshape(-1, 1)
            o_1 = np.array(o_1_r).reshape(-1, 1)

            array_data = data._rearrange_data_v2(object_id, object_label, x_1, y_1, v_x, v_y, h_1, w_1, o_1,
                                                 generate_graph=generate_graph,
                                                 trajectory_image=trajectory_image)

            open_file_mode = 'wb' if first_call else 'ab'

            first_call = False

            with open(os.path.join(data.dir_name, data_file_name), open_file_mode) as datafile:
                np.savetxt(datafile, array_data, fmt="%.2f", delimiter=",")

            # Generate augmented positions
            for i in range(data.number_of_augmented_data_per_raw_data):
                x_a = np.array([random.randint(min(x[0], x[1]), max(x[0], x[1]))
                                for x in np.array([x_1_r, x_2_r]).astype(int).T])
                y_a = np.array([random.randint(min(y[0], y[1]), max(y[0], y[1]))
                                for y in np.array([y_1_r, y_2_r]).astype(int).T])
                h_a = np.array([random.randint(min(h[0], h[1]), max(h[0], h[1]))
                                for h in np.array([h_1_r, h_2_r]).astype(int).T])
                w_a = np.array([random.randint(min(w[0], w[1]), max(w[0], w[1]))
                                for w in np.array([w_1_r, w_2_r]).astype(int).T])
                o_a = np.array([random.randint(min(o[0], o[1]), max(o[0], o[1]))
                                for o in np.array([o_1_r, o_2_r]).astype(int).T])

                x_a_1 = x_a[:-1]
                x_a_2 = x_a[1:]
                y_a_1 = y_a[:-1]
                y_a_2 = y_a[1:]
                h_a_1 = h_a[:-1]
                w_a_1 = w_a[:-1]
                o_a_1 = o_a[:-1]

                v_x_a = dd.video_data_fps * (x_a_2 - x_a_1) / (f_2[:-1] - f_1[:-1])
                v_y_a = dd.video_data_fps * (y_a_2 - y_a_1) / (f_2[:-1] - f_1[:-1])

                # Concatenate data
                x_a_1 = np.array(x_a_1).reshape(-1, 1)
                y_a_1 = np.array(y_a_1).reshape(-1, 1)
                v_x_a = np.array(v_x_a).reshape(-1, 1)
                v_y_a = np.array(v_y_a).reshape(-1, 1)
                h_a_1 = np.array(h_a_1).reshape(-1, 1)
                w_a_1 = np.array(w_a_1).reshape(-1, 1)
                o_a_1 = np.array(o_a_1).reshape(-1, 1)

                #augmented_object_id = int(str(data.augmented_object_id_code) + str(object_id) + str(i))
                augmented_object_id = object_id

                array_data = data._rearrange_data_v2(augmented_object_id, object_label, x_a_1, y_a_1, v_x_a, v_y_a,
                                                     h_a_1, w_a_1, o_a_1,
                                                     generate_graph=False,
                                                     trajectory_image=trajectory_image)

                with open(os.path.join(data.dir_name, data_file_name), 'ab') as datafile:
                    np.savetxt(datafile, array_data, fmt="%.2f", delimiter=",")

    # Save and Show the trajectory image
    if generate_graph:
        trajectory_image_name = directory + '/' + data_type_name + '_trajectories.pdf'
        trajectory_image.save_image(os.path.join(data.dir_name, trajectory_image_name))
    if generate_graph and show_graph:
        trajectory_image.show_image()

    # Print finishing message
    print("                                               ---> Done!")
    print("=========================================================")

    return data_file_name
