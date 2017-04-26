__author__ = 'ubriela'

import numpy as np
from collections import defaultdict, Counter
import scipy.stats as stats
import math
from Params import Params

DELIM = "\t"


"""
compute stats of a dataset
"""
def readCheckins(param):
    """
    :param param:
    :return: locs, users, locDict
    """
    min_x, min_y, max_x, max_y = 90, 180, -90, -180
    data = np.loadtxt(param.dataset, dtype='str', delimiter=DELIM)

    user_d, loc_d, locDict = defaultdict(Counter), defaultdict(Counter), {}
    for i in range(data.shape[0]):
        userId, locId, lat, lon = str(data[i][0]), str(data[i][1]), float(data[i][2]), float(data[i][3])
        user_d[userId].update([locId])
        loc_d[locId].update([userId])
        locDict[locId] = (lat, lon)
        min_x, min_y = min(min_x, lat), min(min_y, lon)
        max_x, max_y = max(max_x, lat), max(max_y, lon)

    print "Dataset ", param.dataset
    print "Number of locations ", len(loc_d)
    print "Number of users ", len(user_d)
    print "MBR ", min_x, min_y, max_x, max_y

    return loc_d, user_d, locDict

def entropyStats(locs):
    """
    Compute entropy statistics
    :param locs:
    :return:
    """
    entropies = []
    for c in locs.itervalues():
        if len(c.values()) > 1:
            entropies.append(stats.entropy(c.values(), base=Params.base))
    print "entropy max/avg", max(entropies), np.average(entropies)

def otherStats(users, locs):
    """
    :param users:
    :param locs:
    :return: maxC, maxM
    """
    maxC = 0  # maximum number of visits of a user to a location
    for c in locs.itervalues():
        maxC = max(maxC, c.most_common(1)[0][1])

    maxM = 0 # maximum number of locations visited by a user
    for c in users.itervalues():
        maxM = max(maxM, len(c))

    print "maxC, maxM", maxC, maxM
    return maxC, maxM



# dict_orig = {1:{1:2,2:3}, 2:{1:3,3:1}}
# dict_new = transform(dict_orig)
# print dict_new

"""
U is the number of users
"""
def calculateGridSize(U, eps, sens):
    return max(4,int(math.sqrt(Params.theta * U/math.exp((-np.log(1-Params.p_sigma) * sens)/(eps * Params.gamma)))))


def cellId2Coord(cellId, p):
    """
    Convert from cell id to lat/lon
    :param cellId:
    :param p:
    :return:
    """
    lat_idx = cellId/p.m
    lon_idx = cellId - lat_idx * p.m
    lat = float(lat_idx)/p.m * (p.x_max - p.x_min) + p.x_min
    lon = float(lon_idx)/p.m * (p.y_max - p.y_min) + p.y_min
    return (lat, lon)

def coord2CellId(point, p):
    """
    Convert from lat/lon to cell id
    :param point:
    :param p:
    :return:
    """
    lat, lon = point[0], point[1]
    lat_idx = int((lat - p.x_min) / (p.x_max - p.x_min) * p.m)
    lon_idx = int((lon - p.y_min) / (p.y_max - p.y_min) * p.m)
    cellId = lat_idx * p.m + lon_idx
    return cellId

"""
This function throws data points into an equal-size grid and computes aggregated
statistics associated with each grid cell
"""
def cellStats(p, sens=1):
    # calculate grid granularity

    if p.VARYING_GRID_SIZE:
        p.m = calculateGridSize(len(p.users), p.eps, sens)
        print len(p.users), p.eps, sens, p.m

    cells = defaultdict(Counter)

    for lid in p.locs.keys():
        if not p.locDict.get(lid): print "not exist", lid
        # lat, lon = p.locDict.get(lid)
        # lat_idx = int((lat - p.x_min)/(p.x_max - p.x_min) * p.m)
        # lon_idx = int((lon - p.y_min)/(p.y_max - p.y_min) * p.m)
        # cellId = lat_idx * p.m + lon_idx
        cellId = coord2CellId(p.locDict.get(lid), p)
        cells[cellId].update(p.locs[lid])

    return cells

