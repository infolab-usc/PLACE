__author__ = 'ubriela'

from collections import defaultdict, Counter
import random
import math
import numpy as np
import math
from Params import Params
import scipy.stats as stats
def CEps2Str(C, eps):
    return "C" + str(C) + "_eps" + str(eps)

def getSmoothSensitivityFile(C, eps):
    return "../output/smoothsensitivity/" + CEps2Str (C, eps) + ".txt"

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
            # sampledLocIds = random.sample(list(locs.keys()), M)
            sampledLocs = Counter([(lid, locs[lid]) for lid in random.sample(list(locs.keys()), M)])
            # for lid in sampledLocIds:
            #     sampledLocs[lid] = locs[lid]
            sampledUsers[uid] = sampledLocs
    return sampledUsers

def randomEntropy(n):
    """
    Capture the degree of predictability of the users' whereabouts if each location
    is visited with equal probability
    :param N: number of distinct locations visited by the user
    :return:
    """
    return math.log(n, Params.base)

def temporalUncorrelatedEntropy(pk):
    """
    characterize the heterogeneity of visitation patterns
    :param pk:
    :return:
    """
    return stats.entropy(pk, base=Params.base)

def actualEntropy(pk):
    """
    depends not only on the frequency of visitation, but also the order in which the nodes
    were visited and the time spent at each location, thus capturing the full spatiotemporal
    order present in a person's mobility pattern.
    :param pk:
    :return:
    """
    # p(T') is the probability of finding a particular time-ordered subsequence T' in the trajectory T

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