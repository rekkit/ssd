import copy
import numpy as np

def convert_coordinates(x, conversion_type, start_index=None):
    # get the start index. If it's None, we assume that the starting index is 0.
    start_index = 0 if start_index is None else start_index

    # copy the input and perform the conversion
    res = copy.deepcopy(x)

    if conversion_type == "centroid2corners":
        res[..., start_index] = x[..., start_index] - x[..., start_index + 2] / 2  # x_min = cx - w / 2
        res[..., start_index + 1] = x[..., start_index + 1] - x[..., start_index + 3] / 2  # y_min = cy - h / 2
        res[..., start_index + 2] = res[..., start_index] + x[..., start_index + 2]  # x_max = x_min + w
        res[..., start_index + 3] = res[..., start_index + 1] + x[..., start_index + 3]  # y_max = y_min + h
    elif conversion_type == "centroid2minmax":
        res[..., start_index] = x[..., start_index] - x[..., start_index + 2] / 2  # x_min = cx - w / 2
        res[..., start_index + 1] = res[..., start_index] + x[..., start_index + 2]  # x_max = x_min + w
        res[..., start_index + 2] = x[..., start_index + 1] - x[..., start_index + 3] / 2  # y_min = cy - h / 2
        res[..., start_index + 3] = res[..., start_index + 2] + x[..., start_index + 3]  # y_max = y_min + h
    elif conversion_type == "corners2centroid":
        res[..., start_index] = (x[..., start_index] + x[..., start_index + 2]) / 2  # cx = (x_min + x_max) / 2
        res[..., start_index + 1] = (x[..., start_index + 1] + x[..., start_index + 3]) / 2  # cy = (y_min + y_max) / 2
        res[..., start_index + 2] = x[..., start_index + 2] - x[..., start_index]  # w = x_max - x_min
        res[..., start_index + 3] = x[..., start_index + 3] - x[..., start_index + 1]  # h = y_max - y_min
    elif conversion_type == "minmax2centroid":
        res[..., start_index] = (x[..., start_index] + x[..., start_index + 1]) / 2  # cx = (x_min + x_max) / 2
        res[..., start_index + 1] = (x[..., start_index + 2] + x[..., start_index + 3]) / 2  # cy = (y_min + y_max) / 2
        res[..., start_index + 2] = x[..., start_index + 1] - x[..., start_index]  # w = x_max - x_min
        res[..., start_index + 3] = x[..., start_index + 3] - x[..., start_index + 2]  # h = y_max - y_min
    elif conversion_type in ["minmax2corners", "corners2minmax"]:
        # in this case we just switch the first and second indices since x_min and y_max are in the same position
        res[..., start_index + 1], res[..., start_index + 2] = x[..., start_index + 2], x[..., start_index + 1]
    else:
        raise ValueError(
            "The parameter conversion type must be one of 'centroid2corners', 'centroid2minmax', 'corners2centroid', "
            "'minmax2centroid', 'minmax2corners' or 'corners2minmax' but the value passed is {}".format(
                conversion_type
            )
        )

    return res
