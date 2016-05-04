__author__ = 'ubriela'
import math
import logging
import time
import sys
import numpy as np
from multiprocessing import Pool
from Differential import Differential
from Kd_standard import Kd_standard
from Quad_standard import Quad_standard
from LEParser import read_checkins
from KExp import KExp

from Params import Params

sys.path.append('/Users/ubriela/Dropbox/_USC/_Research/_Crowdsourcing/_Privacy/PSD/src/icde12')

# eps_list = [1]
# eps_list = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
eps_list = [20.0]

# seed_list = [9110]
seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

# C_list = [1]
C_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# C_list = [1,5,10,15,20,15,30,35,40,45]

M_list = [1,2,3,4,5,6,7,8,9,10]

"""
Shannon entropy of two values
"""
def shanon_entropy(f1,f2):
    p1 = float(f1)/(f1+f2)
    p2 = float(f2)/(f1+f2)
    return -(p1 * np.log(p1) + p2 * np.log(p2))

# print shanon_entropy(1,2)
# print shanon_entropy(2,2)
# print shanon_entropy(1,1)
# print shanon_entropy(2,3)
# print shanon_entropy(1,0)


"""
Shannon entropy a list of frequencies
"""
def shanon_entropy_list(l):
    if len(l) == 1:
        return 0
    total = 0
    s = sum(l)
    for v in l:
        c = float(v) / s
        total = total - c * np.log(c)
    return total

# print shanon_entropy_list([2,1,1])

"""
sensitivity of shannon entropy
"""
def sensitivity(N, p_max):
    list = [(1-p_max)/(N-1) for i in range(0, N-1)]
    list.append(p_max)
    entropy = p_max/(1-p_max) * shanon_entropy_list(list)
    const = 1.0/(1-p_max)*shanon_entropy(p_max, 1-p_max)
    # print const, entropy
    return max(entropy, const)

def sensitivity_add(N, p_max):
    list = [(1-p_max)/(N-1) for i in range(0, N-1)]
    list.append(p_max)
    entropy = p_max * shanon_entropy_list(list)
    if p_max <= 0.5:
        const = shanon_entropy(p_max, 1-p_max)
    else:
        const = np.log(2)
    # print const, entropy
    return entropy, const, max(entropy, const)


def sensitivity_c(C, N):
    return C * math.log(N) / (N + C)


def sensitivity_cn(C):
    # find x in range(1,max(e^2,C)]
    # threshold = 0.001
    low = 1.0
    high = max(np.exp(2), C)
    found = False
    while not found:
        x = int((low + high)/2.0)
        if x == low or x == high:
            # fl = low * math.log(low) - low - C
            # fh = high * math.log(high) - high - C
            # if fl < fh:
            #     x = low
            # else:
            #     x = high
            x == low
            break
        f = x * math.log(x) - x - C
        if f < 0:
            low = x
        else:
            high = x

    return (C + 0.0)/x


def vary_n(n):
    return n*(1-np.log(n))

"""
actual sensitivity of shannon entropy
"""
def actual_sensitivity(l):
    e = shanon_entropy_list(l)
    min_e = 1000000
    max_e = 0

    # print "xxx", l

    for i in range(len(l)):
        removed_l = [l[j] for j in range(i) + range(i+1, len(l))]
        tmp_e = shanon_entropy_list(removed_l)
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
    N = 50
    # for C in range(1,100,1):
        # print C, '\t', sensitivity_cn(C)
        # print C-C*np.log(C)+10
    # for n in range(1,100,1):
    #     print vary_n(n)
    # for P in np.arange(0.02,0.21,0.02):
    #     tuple = sensitivity_add(N, P)
    #     print P, '\t', tuple[0], '\t', tuple[1], '\t', tuple[2]
    # for N in np.arange(1,100000,1000):
    #     tuple = sensitivity_add(N, P)
    #     print N, '\t', tuple[0], '\t', tuple[1], '\t', tuple[2]
    for N in np.arange(1,100):
        print N, '\t', sensitivity_c(C, N)


    # p_max = 0.1
    # for N in [2**i for i in np.arange(6,16,1)]:
    #     print N, '\t', sensitivity(N, p_max)

differ = Differential(1000)

"""
add Laplace noise to shannon entropy
"""
def noisyEntropy(count, sens, epsilon):

    if epsilon < 10 ** (-6):
        return count
    else:
        return count + differ.getNoise(sens, epsilon)

"""
root mean square error
"""
def rmse(actual, noisy):
    noisy_vals = []
    actual_vals = []
    # print len(actual), actual
    # print len(noisy), noisy
    for lid in noisy.keys():
        noisy_vals.append(noisy.get(lid))
        if not actual.has_key(lid):
            print lid
        actual_vals.append(actual.get(lid))

    # print len(actual_vals), actual_vals
    # print len(noisy_vals), noisy_vals
    return np.sqrt(((np.array(actual_vals) - np.array(noisy_vals)) ** 2).mean())

"""
mean relative error
"""
def mre(actual, noisy):
    noisy_vals = []
    actual_vals = []
    for lid in noisy.keys():
        noisy_vals.append(noisy.get(lid))
        actual_vals.append(actual.get(lid))

    absErr = np.abs(np.array(actual_vals) - np.array(noisy_vals))
    idx_nonzero = np.where(np.array(actual_vals) != 0)
    absErr_nonzero = absErr[idx_nonzero]
    true_nonzero = np.array(actual_vals)[idx_nonzero]
    relErr = absErr_nonzero / true_nonzero
    return relErr.mean()

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

            E_limit = [shanon_entropy_list(cut_list(locs[key].values(), C)) for key in locs.keys() if len(locs[key].values()) >= p.K]

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

"""
compute actual shannon entropy
"""
def actual_entropy(locs):
    E_actual = {}
    for lid in locs.keys():
        # print lid, shanon_entropy_list(locs[lid].values())
        E_actual[lid] = shanon_entropy_list(locs[lid].values())
    print "average entropy", np.average([e for e in E_actual.values() if e > 0])
    print "max entropy", max([e for e in E_actual.values() if e > 0])
    print "min entropy", min([e for e in E_actual.values() if e > 0])
    print "variance entropy", np.var([e for e in E_actual.values() if e > 0])
    return E_actual

def evalActualSensitivity(p):

    # compute C, f_total
    C = [max(p.locs[key].values()) for key in p.locs.keys() if len(p.locs[key].values()) >= p.K]
    f_total = [sum(p.locs[key].values()) for key in p.locs.keys() if len(p.locs[key].values()) >= p.K]

    # global sensitivity
    sens = [sensitivity(len(C), float(C[k]) / f_total[k]) for k in range(len(C))]

    # actual sensitivity
    sens_a = [actual_sensitivity(p.locs[key].values()) for key in p.locs.keys() if len(p.locs[key].values()) >= p.K]

    print np.mean([sens[i]/sens_a[i] for i in range(len(sens))])
    # print rmse(sens, sens_a)


def data_readin(p):
    """Read in spatial data and initialize global variables."""
    p.select_dataset()
    # data = np.genfromtxt(p.dataset, unpack=True)
    p.locs, p.C, p.f_total, p.users, p.map_locs = read_checkins(p.dataset)

    data = np.ndarray(shape=(2,len(p.users)))
    userids = p.users.keys()
    for i in range (len(p.users)):
        loc_id = int(p.users[userids[i]].keys()[0])  # first loc_id
        data[0][i] = p.map_locs[loc_id][0]
        data[1][i] = p.map_locs[loc_id][1]

    p.NDIM, p.NDATA = data.shape[0], data.shape[1]
    Params.NDIM, Params.NDATA = p.NDIM, p.NDATA
    p.LOW, p.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)
    Params.LOW, Params.HIGH = p.LOW, p.HIGH
    logging.debug(data.shape)
    logging.debug(p.LOW)
    logging.debug(p.HIGH)
    return data


"""
For each user randomly select L locations
"""
def samplingL(user_locs, L):
    sampled_user_locs = {}
    for uid in user_locs.keys():
        locs = user_locs[uid]
        if len(locs) <= L:
            sampled_user_locs[uid] = locs
        else: # sampling
            new_locs = {}
            for i in range(L):
                v = locs.popitem()
                new_locs[v[0]] = v[1]
            sampled_user_locs[uid] = new_locs

    return sampled_user_locs

"""
translate from dict1 to dict_new
"""
def transform(dict_orig):
    dict_new = {}
    for id_orig in dict_orig.keys():
        val_orig = dict_orig[id_orig] # {id, frequency}
        for id_new in val_orig.keys():
            if dict_new.has_key(id_new):
                dict_new[id_new][id_orig] = val_orig[id_new]
            else:
                dict_new[id_new] = {id_orig : val_orig[id_new]}

    return dict_new


# dict_orig = {1:{1:2,2:3}, 2:{1:3,3:1}}
# dict_new = transform(dict_orig)
# print dict_new

"""
This function throws data points into an equal-size grid and computes aggregated
statistics associated with each grid cell
"""

def cell_stats(p):
    c_locs = {}
    c_users = {}
    c_map_locs = {}

    for lid in p.locs.keys():
        lat, lon = p.map_locs.get(lid)
        lat_idx = int((lat - p.x_min)/(p.x_max - p.x_min) * p.m)
        lon_idx = int((lon - p.y_min)/(p.y_max - p.y_min) * p.m)
        c_lid = lat_idx * p.m + lon_idx

        if c_locs.has_key(c_lid):
            users_freqs = p.locs.get(lid)
            for uid in users_freqs.keys():
                if c_locs[c_lid].has_key(uid):
                    c_locs[c_lid][uid] = c_locs[c_lid][uid] + users_freqs[uid]
                else:
                    c_locs[c_lid][uid] = users_freqs[uid]
        else:
            c_locs[c_lid] = p.locs.get(lid)

    c_users = transform(c_locs)

    return c_locs, c_users, c_map_locs

"""
replace all values in the list by C if they are larger than C
"""
def cut_list(l, C):
    return [C if v > C else v for v in l]

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
def evalLimitCM(params):
    """
    only consider locations with more than K users
    """
    logging.info("evalLimitCM")
    exp_name = "evalLimitCM"
    methodList = ["RMSE_NA", "RMSE_CN", "RMSE_CA", "MRE_NA", "MRE_CN", "MRE_CA"]

    p = params[0]
    p.C = params[1]
    p.M = params[2]
    E_actual = params[3]

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    users = limit_M(p)
    locs = transform(users)
    print "number of locations after limit M", len(locs)

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            eps = eps_list[i]

            # compute C, f_total
            # f_total = [sum(p.locs[key].values()) for key in p.locs.keys() if len(p.locs[key].values()) >= p.K]

            E_limit = {}
            for lid in locs.keys():
                if len(locs[lid].values()) >= p.K:
                    E_limit[lid] = shanon_entropy_list(cut_list(locs[lid].values(), p.C))
                else:
                    print "xxx"
            # global sensitivity
            # sens = [sensitivity(len(f_total), float(C) / f_total[l]) for l in range(len(f_total))]
            # max_sen = max(sens)
            global_sen = sensitivity_cn(float(p.C)) * p.M

            # print len(E_limit), len(E_actual)

            # noisy entropy is computed from rounded entropy
            E_noisy = {}
            for lid in E_limit.keys():
                E_noisy[lid] = noisyEntropy(E_limit[lid], global_sen, eps)

            res_cube[i, j, 0] = rmse(E_actual, E_noisy)
            res_cube[i, j, 1] = rmse(E_limit, E_noisy)
            res_cube[i, j, 2] = rmse(E_actual, E_limit)
            res_cube[i, j, 3] = mre(E_actual, E_noisy)
            res_cube[i, j, 4] = mre(E_limit, E_noisy)
            res_cube[i, j, 5] = mre(E_actual, E_limit)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_CM_' + str(p.C) + "_" + str(p.M), res_summary, fmt='%.4f\t')


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

            E_actual = actual_entropy(sampled_loc_users, param.K)
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
    metrics = ['_CM_']
    for col in range(6):
        for metric in metrics:
            out = open(p.resdir + exp_name + str(col), 'w')
            c = 0
            for eps in eps_list:
                line = ""
                for var in var_list:
                    fileName = p.resdir + exp_name + metric + str(var)
                    try:
                        thisfile = open(fileName, 'r')
                    except:
                        sys.exit('no input result file!')
                    # if len(thisfile.readlines()) > 0:
                    #     print thisfile.readlines()
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
def exp1():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.C, p.f_total, p.users, p.map_locs = read_checkins("dataset/gowalla_NY.txt")

    c_locs, c_users, c_map_locs = cell_stats(p)

    # for uid in c_users:
    #     print len(c_users.get(uid))

    for lid in c_locs:
        print len(c_locs.get(lid))

    #
    E_actual = actual_entropy(p.locs, p.K)
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

def exp2():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.C, p.f_total, p.users, p.map_locs = read_checkins("dataset/gowalla_NY.txt")

    E_actual = actual_entropy(p.locs)

    # if not E_actual.has_key(32768):
    #     print "xxx"

    p.debug()

    # evalLimitCM((p, 5, 5, E_actual))

    pool = Pool(processes=len(eps_list))
    params = []
    for M in M_list:
        # param = (p, p.C, M, E_actual)
        # evalLimitCM(param)
        params.append((p, p.C, M, E_actual))
    pool.map(evalLimitCM, params)
    pool.join()

    createGnuData(p, "evalLimitCM", C_list)

def exp3():

    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    p.locs, p.C, p.f_total, p.users, p.map_locs = read_checkins("dataset/gowalla_NY.txt")

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

if __name__ == '__main__':
    exp2()