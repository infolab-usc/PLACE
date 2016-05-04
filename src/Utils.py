__author__ = 'ubriela'

import math
import numpy as np

def distance(lat1, lon1, lat2, lon2):
    """
    Distance between two geographical location
    """
    R = 6371  # km
    dLat = math.radians(abs(lat2 - lat1))
    dLon = math.radians(abs(lon2 - lon1))
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.sin(dLon / 2) * math.sin(dLon / 2) * math.cos(lat1) * math.cos(
        lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def is_rect_cover(rect, loc):
    """
    checks if the rectangle covers a point
    [[x_min,y_min],[x_max,y_max]]
    """
    bool_m1 = rect[0, 0] <= loc[0] <= rect[1, 0]
    bool_m2 = rect[0, 1] <= loc[1] <= rect[1, 1]
    bool_m = np.logical_and(bool_m1, bool_m2)
    if bool_m:
        return True
    else:
        return False

def rect_area(rect):
    """
    Geographical coordinates
    """
    return distance(rect[0][0], rect[0][1], rect[0][0], rect[1][1]) * distance(rect[0][0], rect[0][1], rect[1][0], rect[0][1])