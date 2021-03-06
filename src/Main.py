__author__ = 'ubriela'
import math
import logging
import time
import sys
import random
import numpy as np
import copy

import scipy.stats as stats
from Metrics import KLDiv, RMSE, DivCatScore, TopK, FreqCatScore
from Utils import samplingUsers, transformDict, threshold, CEps2Str, noisyEntropy, noisyCount, noisyPoint, round2Grid, distance, euclideanToRadian, perturbedPoint
from LEBounds import globalSensitivy, localSensitivity, precomputeSmoothSensitivity, diversitySensitivity, smoothSensitivity
from Differential import Differential
from LEStats import cellId2Coord, coord2CellId, entropy, randomEntropy
from multiprocessing import Pool
from Kd_standard import Kd_standard
from Quad_standard import Quad_standard
from LEStats import readCheckins
from KExp import KExp

from Params import Params
from collections import defaultdict, Counter
import sys

sys.path.append('/Users/ubriela/Dropbox/_USC/_Research/_Crowdsourcing/_Privacy/PSD/src/icde12')

# eps_list = [0.1, 0.5, 1.0, 5.0, 10.0]

eps_list = [0.1, 0.4, 0.7, 1.0]

# seed_list = [9110]
seed_list = [9110, 4064, 6903]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

# C_list = [1]
# C_list = [1,5,10,15,20,15,30,35,40,45]

# [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
# M_list = [1,2,3,4,5,6,7,8,9,10]

# K_list = [10,20,30,40,50,60,70,80,90,100]

divMetricList = [DivCatScore, TopK, KLDiv, RMSE]
freqMetricList = [FreqCatScore, TopK, KLDiv, RMSE]
def normalizeEntropy(e):
    """
    Post-processing
    :param e:
    :return:
    """
    return min(Params.MAX_ENTROPY, abs(e))

def normalizeDiversity(d):
    """
    Post-processing
    :param d:
    :return:
    """
    return min(Params.MAX_DIVERSITY, abs(d))

def normalizeFrequency(f):
    """
    Post-processing
    :param f:
    :return:
    """
    return min(Params.MAX_C, abs(f))

def perturbeDiversity(p):
    D_noisy = defaultdict()
    sampledUsers = samplingUsers(p.users, p.M)   # truncate M: keep the first M locations' visits
    locs = transformDict(sampledUsers)

    sensitivity = math.log(p.M, np.e)
    for lid, counter in locs.iteritems():
        if len(counter) >= 1:
            D_noisy[lid] = normalizeDiversity(noisyCount(randomEntropy(len(counter)), sensitivity, p.eps, p.seed))
    return D_noisy


def perturbeEntropy(p, ss, method="SS"):
    E_noisy = defaultdict()
    sampledUsers = samplingUsers(p.users, p.M)   # truncate M: keep the first M locations' visits
    locs = transformDict(sampledUsers)

    if method == "SS":
        # smooth sensitivity
        ssList = ss[CEps2Str(p.C, p.eps)]
        for lid, counter in locs.iteritems():
            if len(counter) >= 1:
                limitFreqs = threshold(counter.values(), p.C)
                smoothSens = ssList[min(len(limitFreqs) - 1, len(ssList) - 1)]
                E_noisy[lid] = normalizeEntropy(
                    entropy(limitFreqs) + smoothSens * 2.0 * p.M / p.eps * np.random.laplace(0, 1, 1)[0])

        sensitivity = p.C * p.M
        E_noisy = defaultdict()
        for lid, counter in locs.iteritems():
            if len(counter) >= 1:
                limitFreqs = threshold(counter.values(), p.C)  # thresholding
                noisyFreqs = [normalizeFrequency(noisyCount(freq, sensitivity, p.eps, p.seed)) for freq in limitFreqs]
                E_noisy[lid] = entropy([f for f in noisyFreqs])  # freq >= 0

    return E_noisy

def perturbeCount(p):
    C_noisy = defaultdict()
    differ = Differential(p.seed)
    for lid, loc in p.locDict.iteritems():
        noisyLoc = differ.addPolarNoise(p.eps, loc, p.radius)  # perturbed noisy location
        cellId = coord2CellId(noisyLoc, p)  # obtain cell id from noisy location
        C_noisy[cellId] = C_noisy.get(cellId, 0) + 1
    return C_noisy

"""
Publish location entropy using Smoooth sensitivity
"""
def evalEnt(p, E_actual, ss):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)
    res_cube = np.zeros((len(eps_list), len(seed_list), len(divMetricList)))

    sampledUsers = samplingUsers(p.users, p.M)   # truncate M: keep the first M locations' visits
    locs = transformDict(sampledUsers)

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            # smooth sensitivity
            ssList = ss[CEps2Str(p.C, p.eps)]

            E_noisy = defaultdict()
            for lid, counter in locs.iteritems():
                if len(counter) >= 1:
                    limitFreqs = threshold(counter.values(), p.C)
                    # print len(ssList), len(limitFreqs)
                    smoothSens = ssList[min(len(limitFreqs) - 1, len(ssList) - 1)]
                    E_noisy[lid] = normalizeEntropy(entropy(limitFreqs) + smoothSens * 2.0 * p.M/p.eps * np.random.laplace(0, 1, 1)[0])

            actual, noisy = [], []
            for lid, e in E_actual.iteritems():
                actual.append(e)
                noisy.append(E_noisy.get(lid, Params.DEFAULT_ENTROPY))   # default entropy = 0
            for k in range(len(divMetricList)):
                res_cube[i, j, k] = divMetricList[k](actual, noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + '_M' + str(p.M) + '_C' + str(p.C), res_summary, header="\t".join([f.__name__ for f in divMetricList]), fmt='%.4f\t')

"""
Publish diversity (random entropy) by adding Laplace noise
"""
def evalDiv(p, D_actual):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)
    res_cube = np.zeros((len(eps_list), len(seed_list), len(divMetricList)))

    sampledUsers = samplingUsers(p.users, p.M)   # truncate M: keep the first M locations' visits
    locs = transformDict(sampledUsers)

    sensitivity = p.M # diversitySensitivity(p.M)

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            D_noisy = defaultdict()
            for lid, counter in locs.iteritems():
                if len(counter) >= 1:
                    D_noisy[lid] = randomEntropy(normalizeFrequency(
                        noisyCount(len(counter), sensitivity, p.eps, p.seed))) # add noise to actual count first, then add compute diversity from noisy count

            actual, noisy = [], []
            for lid, d in D_actual.iteritems():
                actual.append(d)
                noisy.append(D_noisy.get(lid, Params.DEFAULT_DIVERSITY))   # default entropy = 0
            for k in range(len(divMetricList)):
                res_cube[i, j, k] = divMetricList[k](actual, noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + '_M' + str(p.M) + '_C' + str(p.C), res_summary, header="\t".join([f.__name__ for f in divMetricList]), fmt='%.4f\t')

"""
Publish diversity (random entropy) using GeoI
"""
def evalDivGeoI(p, D_actual):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)
    res_cube = np.zeros((len(eps_list), len(seed_list), len(divMetricList)))

    sensitivity = p.M # diversitySensitivity(p.M)

    differ = Differential(p.seed)

    u = distance(p.x_min, p.y_min, p.x_max, p.y_min) * 1000.0 / Params.GEOI_GRID_SIZE
    v = distance(p.x_min, p.y_min, p.x_min, p.y_max) * 1000.0 / Params.GEOI_GRID_SIZE
    rad = euclideanToRadian((u, v))
    cell_size = np.array([rad[0], rad[1]])

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            D_noisy = defaultdict(Counter)
            for lid, loc in p.locDict.iteritems():
                eps = p.eps/sensitivity
                noisyLoc = differ.addPolarNoise(eps, loc, p.radius) # perturbed noisy location

                # rounded to grid
                roundedPoint = round2Grid(noisyLoc, cell_size, p.x_min, p.y_min)

                cellId = coord2CellId(roundedPoint, p)  # obtain cell id from noisy location
                D_noisy[cellId].update(p.locs[lid]) # update count(userid/freq)

            actual, noisy = [], []
            for cellId, d in D_actual.iteritems():
                actual.append(d)
                noisy.append(normalizeDiversity(randomEntropy(len(D_noisy.get(cellId, Counter([]))))))   # default entropy = 0
            for k in range(len(divMetricList)):
                res_cube[i, j, k] = divMetricList[k](actual, noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + '_M' + str(p.M) + '_C' + str(p.C), res_summary, header="\t".join([f.__name__ for f in divMetricList]), fmt='%.4f\t')


"""
Publish location entropy by adding noise to each frequency.
This method assumes the number of users checkins to a location can be known to the adversary
"""
def evalBL(p, E_actual):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)

    res_cube = np.zeros((len(eps_list), len(seed_list), len(divMetricList)))

    sampledUsers = samplingUsers(p.users, p.M)   # truncate M: keep the first M locations' visits
    locs = transformDict(sampledUsers)

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            sensitivity = p.C * p.M

            E_noisy = defaultdict()
            for lid, counter in locs.iteritems():
                if len(counter) >= 1:
                    limitFreqs = threshold(counter.values(), p.C)  # thresholding
                    noisyFreqs = [normalizeFrequency(noisyCount(freq, sensitivity, p.eps, p.seed)) for freq in limitFreqs]
                    E_noisy[lid] = entropy([f for f in noisyFreqs]) # freq >= 0

            actual, noisy = [], []
            for lid, e in E_actual.iteritems():
                actual.append(e)
                noisy.append(E_noisy.get(lid, Params.DEFAULT_ENTROPY))   # default entropy = 0
            for k in range(len(divMetricList)):
                res_cube[i, j, k] = divMetricList[k](actual, noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + '_M' + str(p.M) + '_C' + str(p.C), res_summary, header="\t".join([f.__name__ for f in divMetricList]), fmt='%.4f\t')


"""
Add 2d Laplace noise to each location
"""
def evalCountGeoI(p, C_actual):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)

    res_cube = np.zeros((len(eps_list), len(seed_list), len(freqMetricList)))

    differ = Differential(p.seed)

    u = distance(p.x_min, p.y_min, p.x_max, p.y_min) * 1000.0 / Params.GEOI_GRID_SIZE
    v = distance(p.x_min, p.y_min, p.x_min, p.y_max) * 1000.0 / Params.GEOI_GRID_SIZE
    rad = euclideanToRadian((u, v))
    cell_size = np.array([rad[0], rad[1]])

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            C_noisy = defaultdict()
            for lid, loc in p.locDict.iteritems():
                noisyLoc = differ.addPolarNoise(p.eps, loc, p.radius) # perturbed noisy location

                # rounded to grid
                roundedPoint = round2Grid(noisyLoc, cell_size, p.x_min, p.y_min)

                cellId = coord2CellId(roundedPoint, p)  # obtain cell id from noisy location
                C_noisy[cellId] = C_noisy.get(cellId, 0) + 1

            actual, noisy = [], []
            for cellId, c in C_actual.iteritems():
                if c > 0:
                    actual.append(c)
                    noisy.append(C_noisy.get(cellId, Params.DEFAULT_ENTROPY))   # default entropy = 0
            for k in range(len(freqMetricList)):
                res_cube[i, j, k] = freqMetricList[k](actual, noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_m" + str(p.m) + '_r' + str(p.radius), res_summary, header="\t".join([f.__name__ for f in divMetricList]), fmt='%.4f\t')

"""
Add 1d Laplace noise to number of locations in each cell
"""
def evalCountDiff(p, C_actual):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)

    res_cube = np.zeros((len(eps_list), len(seed_list), len(freqMetricList)))

    sensitivity = 1 # counting query

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            C_noisy = dict([(cellId, normalizeFrequency(noisyCount(c, sensitivity, p.eps, p.seed))) for cellId, c in C_actual.iteritems()])

            actual, noisy = [], []
            for cellId, c in C_actual.iteritems():
                if c > 0:
                    actual.append(c)
                    noisy.append(C_noisy.get(cellId, Params.DEFAULT_ENTROPY))   # default entropy = 0
            for k in range(len(freqMetricList)):
                res_cube[i, j, k] = freqMetricList[k](actual, noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_m" + str(p.m), res_summary, header="\t".join([f.__name__ for f in divMetricList]), fmt='%.4f\t')

"""
Compare the runtime of precomputation of smooth sensitivy: with vs. without early termination
"""
def evalSSTime():
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)
    C_list = range(1,11)
    res_cube_with = np.zeros((len(eps_list), len(seed_list), len(C_list)))
    res_cube_without = np.zeros((len(eps_list), len(seed_list), len(C_list)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            eps = eps_list[i]
            for k, C in enumerate(C_list):
                t1 = time.time()
                smoothSensitivity(C, Params.MAX_N, eps, Params.DELTA)
                runtime = time.time() - t1
                res_cube_with[i, j, k] = runtime

                smoothSensitivity(C, Params.MAX_N, eps, Params.DELTA, False)
                runtime = time.time() - t1
                res_cube_without[i, j, k] = runtime

    res_summary = np.average(res_cube_with, axis=1)
    np.savetxt("../output/" + exp_name + "with_C" + str(C), res_summary,
               header="\t".join([str(c) for c in C_list]), fmt='%.4f\t')

    res_summary = np.average(res_cube_without, axis=1)
    np.savetxt("../output/" + exp_name + "without_C" + str(C), res_summary,
               header="\t".join([str(c) for c in C_list]), fmt='%.4f\t')

# evalSSTime()

def testDifferential():
    p = Params(1000)
    p.select_dataset()
    differ = Differential(1000)
    # RTH = (34.020412, -118.289936)
    TS = (40.758890, -73.985100)

    for i in range(100):
        # (x, y) = differ.getPolarNoise(1000000, p.eps)
        # pp = noisyPoint(TS, (x,y))

        pp = differ.addPolarNoise(1.0, TS, 100)


        # u = distance(p.x_min, p.y_min, p.x_max, p.y_min) * 1000.0 / Params.GRID_SIZE
        # v = distance(p.x_min, p.y_min, p.x_min, p.y_max) * 1000.0 / Params.GRID_SIZE
        # rad = euclideanToRadian((u, v))
        # cell_size = np.array([rad[0], rad[1]])
        # roundedPoint = round2Grid(pp, cell_size, p.x_min, p.y_min)


        roundedPoint = pp
        print (str(roundedPoint[0]) + ',' + str(roundedPoint[1]))



testDifferential()