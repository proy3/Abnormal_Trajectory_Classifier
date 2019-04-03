"""
This script tests the interactive abnormal trajectory generator.
"""

import abnormal_data_generation as adg

# Generate abnormal data
abnormal_data = adg.generate_abnormal_data_from_raw_data(overwrite_data=True,
                                                         generate_graph=True, show_graph=True)
abnormal_data = abnormal_data[:,1:]
