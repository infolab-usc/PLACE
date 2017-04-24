__author__ = 'ubriela'
import math
import logging
import time
import sys
import random
import numpy as np
import copy

import scipy.stats as stats
from Metrics import KLDivergence, rmse
from Utils import samplingUsers, transformDict, threshold, entropy, CEps2Str, noisyEntropy
from LEBounds import globalSensitivy, localSensitivity, precomputeSmoothSensitivity, getSmoothSensitivity

from multiprocessing import Pool
from Kd_standard import Kd_standard
from Quad_standard import Quad_standard
from LEStats import readCheckins
from KExp import KExp

from Params import Params
import collections

sys.path.append('/Users/ubriela/Dropbox/_USC/_Research/_Crowdsourcing/_Privacy/PSD/src/icde12')

eps_list = [0.1, 0.5, 1.0, 5.0, 10.0]

seed_list = [9110, 4064, 6903]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

# C_list = [1]
C_list = [1,2,3,4,5,6,7,8,9,10]
# C_list = [1,5,10,15,20,15,30,35,40,45]

# [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
M_list = [1,2,3,4,5,6,7,8,9,10]

K_list = [10,20,30,40,50,60,70,80,90,100]


def actualEntropy(locs):
    """
    Compute actual shannon entropy from a set of locations
    :param locs:
    :return:
    """
    return dict([(lid, entropy(freqs.values())) for lid, freqs in locs.iteritems()])

def evalSS(p, E_actual):
    logging.info("evalSS")
    exp_name = "evalSS"
    methodList = ["RMSE", "KL"]

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    sampledUsers = samplingUsers(p.users, Params.MAX_M)   # truncate M: keep the first M locations' visits
    locs = transformDict(sampledUsers)

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            # smooth sensitivity
            ss = getSmoothSensitivity([p.C], [p.eps])
            ssList = [v * 2 for v in ss[CEps2Str(p.C, p.eps)]]

            E_noisy = {}
            for lid, freqs in locs.iteritems():
                if len(freqs) >= 1:
                    limitFreqs = threshold(freqs, p.C)
                    n = len(limitFreqs) - 1
                    smoothSens = ssList[n]
                    e = entropy(limitFreqs)
                    E_noisy[lid] = noisyEntropy(e, smoothSens, p.eps)
                # else:
                #     E_noisy[lid] = 0
                    # print "!!! few sampledUsers in this location !!!"

            actual, noisy = [], []
            for lid, ne in E_noisy.iteritems():
                noisy.append(ne)
                actual.append(E_actual[lid])
            res_cube[i, j, 0] = KLDivergence(actual, noisy)
            res_cube[i, j, 1] = rmse(actual, noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_' + str(p.C), res_summary, fmt='%.4f\t')


