__author__ = 'ubriela'

from collections import defaultdict, Counter
import random
import math
import numpy as np
import math
from Params import Params
import scipy.stats as stats
from Differential import Differential
import heapq

def topKValues(k, values):
    """
    Return top k largest values and their indices from a list
    :param k:
    :param values:
    :return:
    """

    # h = [(i, v) for i, v in enumerate(values[:k])]
    # heapq.heapify(h)
    # if len(values) > k:
    #     for i, val in enumerate(values[k:]):
    #         heapq.heappushpop(h, (i + k, val))
    # return h


    return heapq.nlargest(k, [(val, i) for i, val in enumerate(values)])

def round2Grid(point, cell_size, x_offset, y_offset):
    """
    Round the coordinates of a point to the points of a grid.
    :param point: The moint to migrate.
    :param cell_size: Size of the grid to round towards (ndarray)
    :return: The migrated point
    """
    xy = np.array([point[0], point[1]]) - np.array([x_offset, y_offset])
    new_xy = np.round(xy / cell_size) * cell_size + np.array([x_offset, y_offset])

    return new_xy

def euclideanToRadian(radian):
    """
    Convert from euclidean scale to radian scale
    :param radian:
    :return:
    """
    return (radian[0] * Params.ONE_KM * 0.001, radian[1] * Params.ONE_KM * 1.2833 * 0.001)

def perturbedPoint(point, p):
    """
    Perturbed point with 2D Laplace noise
    :param point:
    :param p:
    :param eps:
    :return:
    """
    differ = Differential(p.seed)
    (x, y) = differ.getPolarNoise(p.radius, p.eps * p.M)
    pp = noisyPoint(point, (x,y))
    u = distance(p.x_min, p.y_min, p.x_max, p.y_min) * 1000.0 / Params.GRID_SIZE
    v = distance(p.x_min, p.y_min, p.x_min, p.y_max) * 1000.0 / Params.GRID_SIZE
    rad = euclideanToRadian((u, v))
    cell_size = np.array([rad[0], rad[1]])
    roundedPoint = round2Grid(pp, cell_size, p.x_min, p.y_min)
    return roundedPoint
    # print (str(roundedPoint[0]) + ',' + str(roundedPoint[1]))

def noisyPoint(point, noise):
    """
    Add 2D noise to a point
    :param point: actual coordinates
    :param noise: in euclidean distance
    :return:
    """
    coordDelta = euclideanToRadian(noise)
    return (point[0] + coordDelta[0], point[1] + coordDelta[1])

def noisyCount(count, sens, epsilon, seed):
    """
    Add Laplace noise to Shannon entropy
    :param count: actual count
    :param sens: sensitivity
    :param epsilon: privacy loss
    :return:
    """
    differ = Differential(seed)
    if epsilon < Params.MIN_SENSITIVITY / 100:
        return count
    else:
        return count + differ.getNoise(sens, epsilon)

def noisyEntropy(count, sens, epsilon, seed):
    """
    Add Laplace noise to Shannon entropy
    :param count: actual count
    :param sens: sensitivity
    :param epsilon: privacy loss
    :return:
    """
    differ = Differential(seed)
    if epsilon < Params.MIN_SENSITIVITY/100:
        return count
    else:
        return count + differ.getNoise(sens, epsilon)

def CEps2Str(C, eps):
    return "C" + str(C) + "_eps" + str(eps)

def getSmoothSensitivityFile(C, eps):
    return "../output/smooth/" + CEps2Str (C, eps) + ".txt"

def threshold(values, C):
    """
    Threshold all values in the list by C if they are larger than C

    :param values: list of values
    :param C: threshold
    :return:
    """
    return [C if v > C else v for v in values]

def samplingUsers(users, M):
    """
    For each user randomly select M locations
    :param users:
    :param M: maximum number of locations visited by a users
    :return:
    """
    sampledUsers = defaultdict(Counter)
    for uid, locs in users.iteritems():
        if len(locs) <= M:
            sampledUsers[uid] = locs
        else: # sampling
            sampledLocIds = random.sample(list(locs.keys()), M)
            sampledLocs = dict([(lid, locs[lid]) for lid in sampledLocIds])
            sampledUsers[uid] = sampledLocs
    return sampledUsers

"""
convert from origDict to newDict
"""
def transformDict(origDict):
    newDict = defaultdict(Counter)
    for origId, orgCounter in origDict.iteritems():
        for newId, freq in orgCounter.iteritems():
            newDict[newId].update(Counter({origId : freq}))
    return newDict


# print entropy([1, 1000000, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1])

"""
depends not only on the frequency of visitation, but also the order in which the nodes
were visited and the time spent at each location, thus capturing the full spatiotemporal
order present in a person's mobility pattern.
:param pk:
:return:
"""
# def actualEntropy(pk):

    # p(T') is the probability of finding a particular time-ordered subsequence T' in the trajectory T

def distance(lat1, lon1, lat2, lon2):
    """
    Distance between two geographical location (km)
    """
    R = Params.EARTH_RADIUS/1000  # km
    dLat = math.radians(abs(lat2 - lat1))
    dLon = math.radians(abs(lon2 - lon1))
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.sin(dLon / 2) * math.sin(dLon / 2) * math.cos(lat1) * math.cos(
        lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

# print distance(34.020412, -118.289936, 34.021969, -118.279983)

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