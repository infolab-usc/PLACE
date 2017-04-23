__author__ = 'ubriela'
import math
import logging
import time
import sys
import numpy as np
import copy
import random
import scipy.stats as stats
from multiprocessing import Pool
from Differential import Differential
from Kd_standard import Kd_standard
from Quad_standard import Quad_standard
from LEStats import readCheckins
from KExp import KExp

from Params import Params
import collections

sys.path.append('/Users/ubriela/Dropbox/_USC/_Research/_Crowdsourcing/_Privacy/PSD/src/icde12')

# eps_list = [1]
# eps_list = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
eps_list = [0.1, 1, 10, 100]

seed_list = [9110, 4064, 6903]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

# C_list = [1]
C_list = [1,2,3,4,5,6,7,8,9,10]
# C_list = [1,5,10,15,20,15,30,35,40,45]

# [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
M_list = [1,2,3,4,5,6,7,8,9,10]

K_list = [10,20,30,40,50,60,70,80,90,100]

# print stats.entropy(2,3)
# print stats.entropy(2,2)
# print stats.entropy(1,1)
# print stats.entropy(2,3)
# print stats.entropy(1,0)

# print stats.entropy([0.5, 0.05, 0.45])
# print stats.entropy([2,3,2,2,3])
# x2 = stats.entropy([8,8,4])

# print stats.entropy([2,3,4], [4,6,8])
# print stats.entropy([4,6,8], [2,3,4])

# print x1, x2
# x3 = stats.entropy([8,8,8,7,6,5,4,3,2,1])
# x4 = stats.entropy([7,7,7,7,6,5,4,3,2,1])
# x5 = stats.entropy([6,6,6,6,6,5,4,3,2,1])
# x6 = stats.entropy([5,5,5,5,5,5,4,3,2,1])
# x7 = stats.entropy([4,4,4,4,4,4,4,3,2,1])
# x8 = stats.entropy([3,3,3,3,3,3,3,3,2,1])
# x9 = stats.entropy([2,2,2,2,2,2,2,2,2,1])
# x10 = stats.entropy([1,1,1,1,1,1,1,1,1,1])
# print x1<x2<x3<x4<x5<x6<x7<x8<x9<x10


if False:
    l = random.sample(range(1,51), 50)
    print l
    for i in range(50):
        print l[i], "\t", stats.entropy(l) - stats.entropy(l[:i] + l[i+1 :])

"""
sensitivity of shannon entropy
"""
# def sensitivity(N, p_max):
#     list = [(1-p_max)/(N-1) for i in range(0, N-1)]
#     list.append(p_max)
#     first = p_max/(1-p_max) * stats.entropy(list)
#     second = 1.0/(1-p_max)*stats.entropy(p_max, 1-p_max)
#     # print const, entropy
#     return max(first, second)

"""
k is the minimum number of users per locations
"""
def sensitivity_add(C, p_max):
    first = - p_max * p_max * np.log(p_max) - p_max * (1-p_max) * np.log(p_max * (1-p_max)/(C-p_max))
    second = stats.entropy(p_max, 1-p_max)
    # print first, "\t", second
    return first, second, max(first, second)


# def sensitivity_k1(n, C):
#     k0 = np.log((n-1+0.0)/(n-1+C)) + (C + 0.0)/(n-1+C)*np.log(C)
#     k1 = np.log((n+0.0)/(n+C)) + (C + 0.0)/(n+C)*np.log(C)
#     k2 = np.log(1 + 1/(np.exp(np.log(n+1)*np.log(C)/(C-1) + np.log(np.log(C)/(C-1))) + 1))
#     return max(k0,k1,k2)


def sensitivity_h1(n, C):
    p_max = (C+0.0)/n
    first = - p_max * p_max * np.log(p_max) - p_max * (1-p_max) * np.log((1-p_max)/(n-1))
    return first

# c_min = 20
# n_max = 40
# sen_max = 0
# for c in range(1,c_min + 1):
#     for n in range(n_max, n_max + 100):
#         sen_h = sensitivity_h1(n, c)
#         sen_k = sensitivity_k1(n, c)
#         # print sen_h, sen_k, sen_h/sen_k
#         if sen_k > sen_max:
#             sen_max = sen_k
#             print sen_max, c, n

# print sensitivity_add(200, 2.0/200)[2] * 5

def sensitivity_c(C, N):
    return C * math.log(N) / (N + C)


def sensitivity_cn(C):
    # find x in range(1,max(e^2,C)]
    # threshold = 0.001
    low = 1.0
    high = max(np.exp(2), C + 0.0)
    found = False
    while not found:
        x = int((low + high)/2.0)
        if x == low or x == high:
            fl = C*np.log(low)/(C+np.log(low))
            fh = C*np.log(high)/(C+np.log(high))
            return fl if fl > fh else fh
        f = x * math.log(x) - x - C
        if f < 0:
            low = x
        else:
            high = x

    # return (C + 0.0)/x


def vary_n(n):
    return n*(1-np.log(n))

"""
compute actual shannon entropy
"""
def shannonEntropy(locs):
    E_actual = {}
    for lid in locs.keys():
        # print lid, stats.entropy(locs[lid].values())
        E_actual[lid] = stats.entropy(locs[lid].values())
    # print "average entropy", np.average([e for e in E_actual.values() if e > 0])
    # print "max entropy", max([e for e in E_actual.values() if e > 0])
    # print "min entropy", min([e for e in E_actual.values() if e > 0])
    # print "variance entropy", np.var([e for e in E_actual.values() if e > 0])
    return E_actual

"""
actual sensitivity of shannon entropy
"""
def actual_sensitivity(l):
    e = stats.entropy(l)
    min_e = 1000000
    max_e = 0

    # print "xxx", l

    for i in range(len(l)):
        removed_l = [l[j] for j in range(i) + range(i+1, len(l))]
        tmp_e = stats.entropy(removed_l)
        # print tmp_e, removed_l
        max_e = max(tmp_e, max_e)
        min_e = min(tmp_e, min_e)
        # print min_e, max_e

    max_e = max(e, max_e)
    min_e = min(e, min_e)
    return max_e - min_e

# Test entropy sensitivity
if False:
    P = 0.5
    C = 10
    N = 100000
    # for C in range(1,100,1):
    #     print C, '\t', sensitivity_cn(C)
        # print C-C*np.log(C)+10
    # for n in range(1,100,1):
    #     print vary_n(n)
    for P in np.arange(0.02,0.21,0.02):
        tuple = sensitivity_add(N, P)
        print P, '\t', tuple[0], '\t', tuple[1], '\t', tuple[2]
    # for N in np.arange(1,100000,1000):
    #     tuple = sensitivity_add(N, P)
    #     print N, '\t', tuple[0], '\t', tuple[1], '\t', tuple[2]
    # for N in np.arange(1,100):
    #     print N, '\t', sensitivity_c(C, N)


    # p_max = 0.1
    # for N in [2**i for i in np.arange(6,16,1)]:
    #     print N, '\t', sensitivity(N, p_max)




# actual = dict({})
# actual[1] = 1
# actual[2] = 2
# actual[3] = 3
# actual[4] = 4
# noisy = dict({})
# noisy[1] = 2
# noisy[2] = 1
# noisy[3] = 4
# noisy[4] = 3
#
# print edit_distance(actual, noisy)

"""
varying budget
"""
def evalAll(params):
    logging.info("evalAll")
    exp_name = "evalAll"
    methodList = ["RMSE", "MRE"]

    p = params[0]
    eps = params[1]
    E_actual = params[2]

    eps_list = [eps]

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            # compute C, f_total
            C = [max(p.locs[key].values()) for key in p.locs.keys() if len(p.locs[key].values()) >= p.K]
            f_total = [sum(p.locs[key].values()) for key in p.locs.keys() if len(p.locs[key].values()) >= p.K]

            # global sensitivity
            sens = [sensitivity(len(C), float(C[k]) / f_total[k]) for k in range(len(C))]
            max_sen = max(sens)

            # noisy entropy
            E_noisy = [noisyEntropy(E_actual[k], max_sen, p.eps) for k in range(len(E_actual))]

            res_cube[i, j, 0] = rmse(E_actual,E_noisy)
            res_cube[i, j, 1] = mre(E_actual,E_noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_eps_' + str(p.eps), res_summary, fmt='%.4f\t')


"""
varying C
"""
def evalLimitCM2(params):
    """
    only consider locations with more than K users
    """
    p = params[0]
    locs = params[1]
    C = params[2]
    E_actual = params[3]
    L = params[4]

    logging.info("evalLimitCM")
    exp_name = "evalLimitCM2"
    methodList = ["RMSE_NA", "RMSE_CN", "RMSE_CA", "MRE_NA", "MRE_CN", "MRE_CA"]

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            eps = eps_list[i]

            # compute C, f_total
            f_total = [sum(locs[key].values()) for key in locs.keys() if len(locs[key].values()) >= p.K]

            E_limit = [stats.entropy(cut_list(locs[key].values(), C)) for key in locs.keys() if len(locs[key].values()) >= p.K]

            # global sensitivity
            sens = [sensitivity(len(f_total), float(C) / f_total[l]) for l in range(len(f_total))]
            max_sen = max(sens) * L

            # noisy entropy is computed from rounded entropy
            E_noisy = [noisyEntropy(E_limit[l], max_sen, eps) for l in range(len(E_actual))]

            res_cube[i, j, 0] = rmse(E_noisy, E_actual)
            res_cube[i, j, 1] = rmse(E_limit, E_noisy)
            res_cube[i, j, 2] = rmse(E_limit, E_actual)
            res_cube[i, j, 3] = mre(E_noisy, E_actual)
            res_cube[i, j, 4] = mre(E_limit, E_noisy)
            res_cube[i, j, 5] = mre(E_limit, E_actual)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_CM_' + str(C), res_summary, fmt='%.4f\t')

def evalActualSensitivity(p):

    # compute C, f_total
    C = [max(p.locs[key].values()) for key in p.locs.keys() if len(p.locs[key].values()) >= p.K]

    # global sensitivity
    sens = [sensitivity_add(p.C, float((p.C + 0.0) / p.K))[2] for k in range(len(C))]

    # actual sensitivity
    sens_a = [actual_sensitivity(p.locs[key].values()) for key in p.locs.keys() if len(p.locs[key].values()) >= p.K]

    print sens
    print sens_a
    print np.mean([sens[i]/sens_a[i] for i in range(len(sens)) if sens_a[i] > 0])
    # print rmse(sens, sens_a)


def data_readin(p):
    """Read in spatial data and initialize global variables."""
    p.select_dataset()
    # data = np.genfromtxt(p.dataset, unpack=True)
    p.locs, p.C, p.f_total, p.users, p.locDict = readCheckins(p.dataset)

    data = np.ndarray(shape=(2,len(p.users)))
    userids = p.users.keys()
    for i in range (len(p.users)):
        loc_id = int(p.users[userids[i]].keys()[0])  # first loc_id
        data[0][i] = p.locDict[loc_id][0]
        data[1][i] = p.locDict[loc_id][1]

    p.NDIM, p.NDATA = data.shape[0], data.shape[1]
    Params.NDIM, Params.NDATA = p.NDIM, p.NDATA
    p.LOW, p.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)
    Params.LOW, Params.HIGH = p.LOW, p.HIGH
    logging.debug(data.shape)
    logging.debug(p.LOW)
    logging.debug(p.HIGH)
    return data







"""
for users who visits more than M locations,
choose the first M only and throw away the rest
"""
def limit_M(p):
    users = {}

    for uid in p.users.keys():
        # print len(p.users.get(uid))
        if len(p.users.get(uid)) <= p.M:
            users[uid] = p.users.get(uid)
        else:
            # obtain the first M locations
            count = 0
            locs = {}
            for lid, freq in p.users.get(uid).iteritems():
                locs[lid] = freq
                count = count + 1
                if count == p.M:
                    break
            users[uid] = locs

    return users



"""
varying C
"""
def evalLimitC(params):
    logging.info("evalLimit")
    exp_name = "evalLimitC"
    methodList = ["RMSE_NA", "RMSE_CN", "RMSE_CA", "MRE_NA", "MRE_CN", "MRE_CA", "EDIT"]

    p = params[0]
    p.C = params[1]
    E_actual = params[2]

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    users = limit_M(p)
    locs = transform(users)

    # locs = p.locs

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            E_limit = {}
            for lid in locs.keys():
                if len(locs[lid].values()) >= 1:
                    E_limit[lid] = stats.entropy(cut_list(locs[lid].values(), p.C))
                else:
                    print "!!! few users in this location !!!"

            global_sen = sensitivity_cn(p.C) * p.M

            # print len(E_limit), len(E_actual)

            # noisy entropy is computed from rounded entropy
            E_noisy = {}
            for lid in E_limit.keys():
                E_noisy[lid] = noisyEntropy(E_limit[lid], global_sen, p.eps)

            res_cube[i, j, 0] = rmse(E_actual, E_noisy)
            res_cube[i, j, 1] = rmse(E_limit, E_noisy)
            res_cube[i, j, 2] = rmse(E_actual, E_limit)
            res_cube[i, j, 3] = mre(E_actual, E_noisy)
            res_cube[i, j, 4] = mre(E_limit, E_noisy)
            res_cube[i, j, 5] = mre(E_actual, E_limit)
            res_cube[i, j, 6] = 0#edit_distance(E_actual, E_noisy)


    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_' + str(p.C), res_summary, fmt='%.4f\t')


"""
varying M
"""
def evalLimitM(params):
    """
    only consider locations with more than K users
    """
    logging.info("evalLimitM")
    exp_name = "evalLimitM"
    methodList = ["RMSE_NA", "RMSE_CN", "RMSE_CA", "MRE_NA", "MRE_CN", "MRE_CA", "EDIT"]

    p = params[0]
    p.M = params[1]
    E_actual = params[2]

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    users = limit_M(p)
    locs = transform(users)
    print "number of locations after limit M to ", p.M, len(locs)

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            # compute C, f_total
            # f_total = [sum(p.locs[key].values()) for key in p.locs.keys() if len(p.locs[key].values()) >= p.K]

            E_limit = {}
            for lid in locs.keys():
                if len(locs[lid].values()) >= 1:
                    E_limit[lid] = stats.entropy(cut_list(locs[lid].values(), p.C))
                else:
                    print "xxx"

            global_sen = sensitivity_cn(float(p.C)) * p.M

            # noisy entropy is computed from rounded entropy
            E_noisy = {}
            for lid in E_limit.keys():
                E_noisy[lid] = noisyEntropy(E_limit[lid], global_sen, p.eps)

            res_cube[i, j, 0] = rmse(E_actual, E_noisy)
            res_cube[i, j, 1] = rmse(E_limit, E_noisy)
            res_cube[i, j, 2] = rmse(E_actual, E_limit)
            res_cube[i, j, 3] = mre(E_actual, E_noisy)
            res_cube[i, j, 4] = mre(E_limit, E_noisy)
            res_cube[i, j, 5] = mre(E_actual, E_limit)
            res_cube[i, j, 6] = 0#edit_distance(E_actual, E_noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_' + str(p.M), res_summary, fmt='%.4f\t')

"""
varying K
"""
def evalLimitK(params):
    """
    only consider locations with more than K users
    """
    logging.info("evalLimitK")
    exp_name = "evalLimitK"
    methodList = ["RMSE_NA", "RMSE_CN", "RMSE_CA", "MRE_NA", "MRE_CN", "MRE_CA", "RATIO_CELL", "RATIO_LOC"]

    p = params[0]
    global_sen = params[1]
    E_actual = params[2]

    users = limit_M(p)
    locs = transform(users)

    # print "number of locations after limiting M to", p.M, len(locs)
    published_cells = sum(len(locs[lid]) >= p.K for lid in locs.keys())
    published_locs = sum((len(locs[lid]) >= p.K) * len(locs[lid]) for lid in locs.keys())

    valid_cells = sum(len(locs[lid]) >= p.k_min for lid in locs.keys())
    valid_locs = sum((len(locs[lid]) >= p.k_min) * len(locs[lid]) for lid in locs.keys())
    ratio_cell = published_cells * 100.0/valid_cells
    ratio_loc = published_locs * 100.0/valid_locs
    print "ratio of published cells/locations after limiting K to\t", p.K, "\t", ratio_cell, "\t", ratio_loc
    # print "max users per location", max([len(locs[lid]) for lid in locs.keys()])

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            E_limit = {}
            for lid in locs.keys():
                if len(locs[lid]) >= p.K:
                    E_limit[lid] = stats.entropy(cut_list(locs[lid].values(), p.C))
                # else:
                #     E_limit[lid] = 0

            # noisy entropy is computed from rounded entropy
            E_noisy = {}
            for lid in E_limit.keys():
                E_noisy[lid] = noisyEntropy(E_limit[lid], global_sen, p.eps)

            res_cube[i, j, 0] = rmse(E_actual, E_noisy)
            res_cube[i, j, 1] = rmse(E_limit, E_noisy)
            res_cube[i, j, 2] = rmse(E_actual, E_limit)
            res_cube[i, j, 3] = mre(E_actual, E_noisy)
            res_cube[i, j, 4] = mre(E_limit, E_noisy)
            res_cube[i, j, 5] = mre(E_actual, E_limit)
            res_cube[i, j, 6] = ratio_cell
            res_cube[i, j, 7] = ratio_loc

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_' + str(p.K), res_summary, fmt='%.4f\t')

"""
varying KC
"""
def evalLimitKC(params):
    """
    only consider locations with more than K users
    """
    logging.info("evalLimitKC")
    exp_name = "evalLimitKC"
    methodList = ["RMSE_NA", "RMSE_CN", "RMSE_CA", "MRE_NA", "MRE_CN", "MRE_CA", "RATIO_CELL", "RATIO_LOC"]

    p = params[0]
    global_sen = params[1]
    E_actual = params[2]

    users = limit_M(p)
    locs = transform(users)

    published_cells = sum(len(locs[lid]) >= p.K for lid in locs.keys())
    published_locs = sum((len(locs[lid]) >= p.K) * len(locs[lid]) for lid in locs.keys())

    valid_cells = sum(len(locs[lid]) >= p.k_min for lid in locs.keys())
    valid_locs = sum((len(locs[lid]) >= p.k_min) * len(locs[lid]) for lid in locs.keys())
    ratio_cell = published_cells * 100.0/valid_cells
    ratio_loc = published_locs * 100.0/valid_locs
    print "ratio of published cells/locations after limiting K to\t", p.K, "\t", ratio_cell, "\t", ratio_loc

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            E_limit = {}
            for lid in locs.keys():
                if len(locs[lid]) >= p.K:
                    E_limit[lid] = stats.entropy(cut_list(locs[lid].values(), p.C))
                # else:
                #     E_limit[lid] = 0

            # noisy entropy is computed from rounded entropy
            E_noisy = {}
            for lid in E_limit.keys():
                E_noisy[lid] = noisyEntropy(E_limit[lid], global_sen, p.eps)

            res_cube[i, j, 0] = rmse(E_actual, E_noisy)
            res_cube[i, j, 1] = rmse(E_limit, E_noisy)
            res_cube[i, j, 2] = rmse(E_actual, E_limit)
            res_cube[i, j, 3] = mre(E_actual, E_noisy)
            res_cube[i, j, 4] = mre(E_limit, E_noisy)
            res_cube[i, j, 5] = mre(E_actual, E_limit)
            res_cube[i, j, 6] = ratio_cell
            res_cube[i, j, 7] = ratio_loc

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_' + str(p.C), res_summary, fmt='%.4f\t')

"""
varying KM
"""
def evalLimitKM(params):
    """
    only consider locations with more than K users
    """
    logging.info("evalLimitKM")
    exp_name = "evalLimitKM"
    methodList = ["RMSE_NA", "RMSE_CN", "RMSE_CA", "MRE_NA", "MRE_CN", "MRE_CA", "EDIT"]

    p = params[0]
    global_sen = params[1]
    E_actual = params[2]

    users = limit_M(p)
    locs = transform(users)

    print "number of locations after limiting M to ", p.M, len(locs)
    print "number of locations after limiting K to ", p.K, sum(len(locs[lid]) >= p.K for lid in locs.keys())

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            E_limit = {}
            for lid in locs.keys():
                if len(locs[lid]) >= p.K:
                    E_limit[lid] = stats.entropy(cut_list(locs[lid].values(), p.C))
                # else:
                #     E_limit[lid] = 0

            # noisy entropy is computed from rounded entropy
            E_noisy = {}
            for lid in E_limit.keys():
                E_noisy[lid] = noisyEntropy(E_limit[lid], global_sen, p.eps)

            res_cube[i, j, 0] = rmse(E_actual, E_noisy)
            res_cube[i, j, 1] = rmse(E_limit, E_noisy)
            res_cube[i, j, 2] = rmse(E_actual, E_limit)
            res_cube[i, j, 3] = mre(E_actual, E_noisy)
            res_cube[i, j, 4] = mre(E_limit, E_noisy)
            res_cube[i, j, 5] = mre(E_actual, E_limit)
            res_cube[i, j, 6] = 0#edit_distance(E_actual, E_noisy)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_' + str(p.M), res_summary, fmt='%.4f\t')



def evalPSD(data, param):
    global method_list, exp_name
    exp_name = 'evalPSD'
    method_list = ['Kd_standard']

    # Params.maxHeight = 10
    res_cube_abs = np.zeros((len(eps_list), len(seed_list), len(method_list)))
    res_cube_rel = np.zeros((len(eps_list), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            param.Eps = eps_list[i]
            for k in range(len(method_list)):
                param.Seed = seed_list[j]
                if method_list[k] == 'Quad_standard':
                    tree = Quad_standard(data, param)
                elif method_list[k] == 'Kd_standard':
                    tree = Kd_standard(data, param)
                else:
                    logging.error('No such index structure!')
                    sys.exit(1)

            tree.buildIndex()

            with open(param.dataset, 'r') as ins:
                for row in ins:
                    row = row.split()
                    user_id = int(row[0])
                    # loc_id = int(row[4])
                    leaf = tree.leafCover((float(row[2]), float(row[3])))
                    if leaf == None:
                        print (float(row[2]), float(row[3]))
                        continue

                    # update the number of times user_id visits this location
                    if leaf.users.has_key(user_id):
                        leaf.users[user_id] = leaf.users[user_id] + 1
                    else:
                        leaf.users[user_id] = 1


            loc_users = tree.loc_users()
            print len(loc_users), np.mean([len(loc_users[i]) for i in loc_users.keys()])

            user_locs = transform(loc_users)
            print len(user_locs), np.mean([len(user_locs[k]) for k in user_locs.keys()])

            # sampling
            L = 3
            sampled_user_locs = samplingL(user_locs, L)
            print len(user_locs), np.mean([len(sampled_user_locs[k]) for k in sampled_user_locs.keys()])

            sampled_loc_users = transform(sampled_user_locs)
            print sampled_loc_users
            print len(sampled_loc_users), np.mean([len(sampled_loc_users[i]) for i in sampled_loc_users.keys()])

            E_actual = shannonEntropy(sampled_loc_users, param.K)
            print E_actual
            print len(E_actual)

            # evalLimitCM2((param, sampled_loc_users, 3, E_actual, L))

            pool = Pool(processes=len(eps_list))
            params = []
            for C in C_list:
                params.append((param, sampled_loc_users, C, E_actual, L))
            pool.map(evalLimitCM2, params)
            pool.join()


# exp_name + '_CM_' + str(C)
def createGnuData(p, exp_name, var_list):
    """
    Post-processing output files to generate Gnuplot-friendly outcomes
    """
    for col in range(8):
        out = open(p.resdir + exp_name + str(col), 'w')
        c = 0
        for eps in eps_list:
            line = ""
            for var in var_list:
                fileName = p.resdir + exp_name + "_" + str(var)
                try:
                    thisfile = open(fileName, 'r')
                except:
                    sys.exit('no input result file!' + str(fileName))
                line = line + thisfile.readlines()[c].split("\t")[col] + "\t"
                thisfile.close()
            out.write(line + "\n")
            c += 1
        out.close()

def createGnuData2(p, exp_name, var_list):
    """
    Post-processing output files to generate Gnuplot-friendly outcomes
    """
    metrics = ['_eps_']

    for metric in metrics:
        out = open(p.resdir + exp_name + metric, 'w')
        for var in var_list:
            fileName = p.resdir + exp_name + metric + str(var)
            print fileName
            try:
                thisfile = open(fileName, 'r')
            except:
                sys.exit('no input result file!')
            out.write(thisfile.readlines()[0])
            thisfile.close()
        out.close()


"""
compute statistics
"""
def expStats():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.users, p.locDict = readCheckins("dataset/gowalla_NY.txt")

    c_locs, c_users = cellStats(p)

    # for uid in c_users:
    #     print len(c_users.get(uid))

    for lid in c_locs:
        print len(c_locs.get(lid))

    #
    # E_actual = actual_entropy(p.locs, p.K)
    #
    # p.debug()
    #
    # evalAll((p, 10, E_actual))
    #
    # pool = Pool(processes=len(eps_list))
    # params = []
    # for eps in eps_list:
    #     params.append((p, eps, E_actual))
    # pool.map(evalAll, params)
    # pool.join()

    # createGnuData2(p, "evalAll", eps_list)

def expC():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.users, p.locDict = readCheckins(p)
    E_actual = shannonEntropy(p.locs)
    p.debug()

    # pool = Pool(processes=len(eps_list))
    # params = []
    for C in C_list:
        param = (p, C, E_actual)
        evalLimitC(param)
    #     params.append((p, C, E_actual))
    # pool.map(evalLimitC, params)
    # pool.join()

    createGnuData(p, "evalLimitC", C_list)

def expM():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.users, p.locDict = readCheckins(p)
    E_actual = shannonEntropy(p.locs)
    p.debug()

    # pool = Pool(processes=len(eps_list))
    # params = []
    for M in M_list:
        param = (p, M, E_actual)
        evalLimitM(param)
        # params.append((p, M, E_actual))
    # pool.map(evalLimitM, params)
    # pool.join()

    createGnuData(p, "evalLimitM", M_list)

def expK():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.users, p.locDict = readCheckins(p)
    p.debug()

    copy_locs = copy.deepcopy(p.locs)
    copy_locDict = copy.deepcopy(p.locDict)

    # pool = Pool(processes=len(eps_list))
    # params = []
    for K in K_list:
        p.K = K
        global_sen = sensitivity_add(p.C, float(p.C)/p.K)[2] * p.M
        p.locs, p.users = cellStats(p, copy_locs, copy_locDict, global_sen)
        E_actual = shannonEntropy(p.locs)

        param = (p, global_sen, E_actual)
        evalLimitK(param)
        # params.append((p, C, E_actual))
    # pool.map(evalLimitK, params)
    # pool.join()

    createGnuData(p, "evalLimitK", K_list)

def expKC():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.users, p.locDict = readCheckins(p)
    p.debug()

    copy_locs = copy.deepcopy(p.locs)
    copy_locDict = copy.deepcopy(p.locDict)

    pool = Pool(processes=len(eps_list))
    params = []
    for C in C_list:
        p.C = C
        global_sen = sensitivity_add(p.C, float(p.C)/p.K)[2] * p.M
        p.locs, p.users = cellStats(p, copy_locs, copy_locDict, global_sen)
        E_actual = shannonEntropy(p.locs)

        param = (p, global_sen, E_actual)
        evalLimitKC(param)
        # params.append((p, global_sen, E_actual))
    # pool.map(evalLimitK, params)
    # pool.join()

    createGnuData(p, "evalLimitKC", C_list)

def expKM():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.users, p.locDict = readCheckins(p)
    p.debug()

    copy_locs = copy.deepcopy(p.locs)
    copy_locDict = copy.deepcopy(p.locDict)

    pool = Pool(processes=len(eps_list))
    params = []
    for M in M_list:
        p.M = M
        global_sen = sensitivity_add(p.C, float(p.C)/p.K)[2] * p.M
        p.locs, p.users = cellStats(p, copy_locs, copy_locDict, global_sen)
        E_actual = shannonEntropy(p.locs)

        param = (p, global_sen, E_actual)
        evalLimitKM(param)
        # params.append((p, global_sen, E_actual))
    # pool.map(evalLimitK, params)
    # pool.join()

    createGnuData(p, "evalLimitKM", M_list)

def expSensitivity():

    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.users, p.locDict = readCheckins(p)

    evalActualSensitivity(p)

"""
kd-tree experiment
"""
def exp4():

    logging.basicConfig(level=logging.DEBUG, filename='log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    param = Params(1000)
    data = data_readin(param)

    # print "Loc"
    # for lid in param.locs.keys():
    #     print len(param.locs[lid])

    print "var"
    for lid in param.locs.keys():
        users = param.locs[lid]
        print max(users.values())

    # print "User"
    # for uid in param.users.keys():
    #     print len(param.users[uid])


    param.NDIM, param.NDATA = data.shape[0], data.shape[1]
    param.LOW, param.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)

    evalPSD(data, param)

def testNoisyLocation():
    RTH = (34.020412, -118.289936)
    radius = 500.0  # default unit is meters
    eps = np.log(2)
    # l = radius*eps # higher base_eps gives less privacy
    for i in range(100):
        (x, y) = differ.getTwoPlanarNoise(radius, eps)

        print RTH[0] + x * Params.ONE_KM * 0.001, ',', RTH[1] + y * Params.ONE_KM*1.2833*0.001


if __name__ == '__main__':
    testNoisyLocation()
    # expStats()
    # expSensitivity()
    # expC()
    # expM()
    # expK()
    # expKC()
    # expKM()