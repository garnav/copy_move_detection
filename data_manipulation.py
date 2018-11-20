# data_manipulation.py
# Zhao Shen, Arun Pidugu, Arnav Ghosh
# 19th Nov. 2018

import os
import numpy as np

# TODO Account for no forgeries
# could be used for just the patches themselves as a baseline (later)
# TODO: is 1 or 0 more similar
''' Expects each ROI example to have 4 Dimensions:
        1st Dim: number of ROIs
        2nd, 3rd, 4th: height, width and channels of ROI features

    Expects labels to be in a json file, as follows:
        {filename : [i, j], ...} where filename --> file containng ROI Features
                                       i, j --> indices of ROI features that are duplicates
                                                where i is the original, j is the forgery '''
def load_data(x_data_path, y_data_path):
    data_x = None
    data_y = None
    rois_per_example = None
    all_files = os.listdir(x_data_path)
    with open(y_data_path, 'r') as jsonFile:
        file_label_map = json.load(jsonFile)
        for i, file in enumerate(all_files):
            x = np.load(file)

            # assuming shape is 4D
            assert len(x.shape) == 4

            if data_x is None:
                data_x = np.zeros((len(all_files), *x.shape))
                rois_per_example = x.shape[0]
                data_y = np.zeros((len(all_files), (rois_per_example * (rois_per_example - 1)) ))

            data_x[i, ] = x
            duplicate_indices = file_label_map[file]
            duplicate_indices.sort()

            label = []
            for i in range(t):
                for j in range(i + 1, t):
                    # after sorting, j will always equal the second index when
                    # i equals the first index
                    if i == duplicate_indices[0] and j == duplicate_indices[1]:
                        label.append(1)
                    else:
                        label.append(0)

            # TODO: use one hot vector?
            data_y[i, ] = np.array(label)

    return data_x, data_y
