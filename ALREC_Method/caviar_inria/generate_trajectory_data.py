"""
This script generates the normal and abnormal trajectory data.
"""
import warnings
from caviar_inria import extract_real_data as cdata


warnings.simplefilter(action='ignore', category=FutureWarning)

generate_trajectory_samples_with_features = False

if generate_trajectory_samples_with_features:
    # Extract trajectories and export data to array
    cdata.extract_augment_and_export_data(is_normal=True,
                                          generate_graph=False,
                                          show_graph=False)

    cdata.extract_augment_and_export_data(is_normal=False,
                                          generate_graph=False,
                                          show_graph=False)
else:
    # Extract trajectories and export data to array
    cdata.extract_augment_and_export_positions_data(is_normal=True,
                                                    generate_graph=False,
                                                    show_graph=False)

    cdata.extract_augment_and_export_positions_data(is_normal=False,
                                                    generate_graph=False,
                                                    show_graph=False)
