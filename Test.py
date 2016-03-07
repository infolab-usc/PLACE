__author__ = 'ubriela'
import math
import logging
import time
import sys
import numpy as np
from multiprocessing import Pool
from Differential import Differential

from LEParser import read_checkins

from Params import Params

# eps_list = [1]
eps_list = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

# seed_list = [9110]
seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

# f_max_list = [1]
f_max_list = [1,2,3,4,5,6,7,8,9,10]
# f_max_list = [1,5,10,15,20,15,30,35,40,45]

"""
Shannon entropy of two values
"""
def shanon_entropy(f1,f2):
    p1 = float(f1)/(f1+f2)
    p2 = float(f2)/(f1+f2)
    return -(p1 * np.log(p1)/np.log(2) + p2 * np.log(p2)/np.log(2))

print shanon_entropy(1,2)
# print shanon_entropy(2,2)
# print shanon_entropy(1,1)
# print shanon_entropy(2,3)
# print shanon_entropy(1,0)


"""
Shannon entropy a list of frequencies
"""
def shanon_entropy_list(l):
    total = 0
    s = sum(l)
    for v in l:
        c = float(v) / s
        total = total - c * np.log(c)/np.log(2)
    return total

print shanon_entropy_list([2,1,1])

"""
sensitivity of shannon entropy
"""
def sensitivity(N, p_max):
    list = [(1-p_max)/(N-1) for i in range(0, N-1)]
    list.append(p_max)
    entropy = p_max/(1-p_max) * shanon_entropy_list(list)
    const = 1.0/(1-p_max)*shanon_entropy(p_max, 1-p_max)
    # print N, p_max, const + entropy
    return entropy + const


# Test entropy sensitivity
if False:
    N = 100
    for p_max in np.arange(0.01,0.2,0.02):
        print p_max, '\t', sensitivity(N, p_max)


    p_max = 0.1
    for N in [2**i for i in np.arange(6,16,1)]:
        print N, '\t', sensitivity(N, p_max)

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
    return np.sqrt(((np.array(actual) - np.array(noisy)) ** 2).mean())

"""
mean relative error
"""
def mre(actual, noisy):
    absErr = np.abs(np.array(actual) - np.array(noisy))
    idx_nonzero = np.where(np.array(actual) != 0)
    absErr_nonzero = absErr[idx_nonzero]
    true_nonzero = np.array(actual)[idx_nonzero]
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
    E_a = params[2]

    eps_list = [eps]

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            p.eps = eps_list[i]

            # compute f_max, f_total
            f_max = [max(p.d_locs[key].values()) for key in p.d_locs.keys() if len(p.d_locs[key].values()) >= p.K]
            f_total = [sum(p.d_locs[key].values()) for key in p.d_locs.keys() if len(p.d_locs[key].values()) >= p.K]

            # global sensitivity
            sens = [sensitivity(len(f_max), float(f_max[k]) / f_total[k]) for k in range(len(f_max))]
            max_sen = max(sens)

            # noisy entropy
            E_n = [noisyEntropy(E_a[k], max_sen, p.eps) for k in range(len(E_a))]

            res_cube[i, j, 0] = rmse(E_a,E_n)
            res_cube[i, j, 1] = mre(E_a,E_n)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_eps_' + str(p.eps), res_summary, fmt='%.4f\t')

"""
replace all values in the list by f_max if they are larger than f_max
"""
def cut_list(l, f_max):
    return [f_max if v > f_max else v for v in l]

"""
varying K
"""
def evalCutFreq(params):
    """
    only consider locations with more than K users
    """
    logging.info("evalCutFreq")
    exp_name = "evalCutFreq"
    methodList = ["RMSE_NA", "RMSE_CN", "RMSE_CA", "MRE_NA", "MRE_CN", "MRE_CA"]

    p = params[0]
    f_max = params[1]
    E_a = params[2]

    res_cube = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            p.seed = seed_list[j]
            eps = eps_list[i]

            # compute f_max, f_total
            f_total = [sum(p.d_locs[key].values()) for key in p.d_locs.keys() if len(p.d_locs[key].values()) >= p.K]

            E_c = [shanon_entropy_list(cut_list(p.d_locs[key].values(), f_max)) for key in p.d_locs.keys() if len(p.d_locs[key].values()) >= p.K]

            # global sensitivity
            sens = [sensitivity(len(f_total), float(f_max) / f_total[l]) for l in range(len(f_total))]
            max_sen = max(sens)

            # noisy entropy is computed from rounded entropy
            E_n = [noisyEntropy(E_c[l], max_sen, eps) for l in range(len(E_a))]

            res_cube[i, j, 0] = rmse(E_n, E_a)
            res_cube[i, j, 1] = rmse(E_c, E_n)
            res_cube[i, j, 2] = rmse(E_c, E_a)
            res_cube[i, j, 3] = mre(E_n, E_a)
            res_cube[i, j, 4] = mre(E_c, E_n)
            res_cube[i, j, 5] = mre(E_c, E_a)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + exp_name + '_f_max_' + str(f_max), res_summary, fmt='%.4f\t')

"""
compute actual shannon entropy
"""
def actual_entropy(p):
    E_a = [shanon_entropy_list(p.d_locs[key].values()) for key in p.d_locs.keys() if len(p.d_locs[key].values()) >= p.K]
    print "average entropy", np.average([e for e in E_a if e > 0])
    print "max entropy", max([e for e in E_a if e > 0])
    print "min entropy", min([e for e in E_a if e > 0])
    print "variance entropy", np.var([e for e in E_a if e > 0])
    return E_a

# exp_name + '_f_max_' + str(f_max)
def createGnuData(p, exp_name, var_list):
    """
    Post-processing output files to generate Gnuplot-friendly outcomes
    """
    metrics = ['_f_max_']
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

def exp1():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    # p.d_locs, p.f_max, p.f_total, p.users = read_checkins("dataset/gowalla_NY.txt")
    #
    # E_a = actual_entropy(p)
    #
    # p.debug()
    #
    # evalAll((p, 10, E_a))
    #
    # pool = Pool(processes=len(eps_list))
    # params = []
    # for eps in eps_list:
    #     params.append((p, eps, E_a))
    # pool.map(evalAll, params)
    # pool.join()

    createGnuData2(p, "evalAll", eps_list)

def exp2():
    logging.basicConfig(level=logging.DEBUG, filename='./debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)

    p.select_dataset()

    # p.d_locs, p.f_max, p.f_total, p.users = read_checkins("dataset/gowalla_NY.txt")
    #
    # E_a = actual_entropy(p)
    #
    # p.debug()
    #
    # evalCutFreq((p, 10, E_a))
    #
    # pool = Pool(processes=len(eps_list))
    # params = []
    # for f_max in f_max_list:
    #     params.append((p, f_max, E_a))
    # pool.map(evalCutFreq, params)
    # pool.join()

    createGnuData(p, "evalCutFreq", f_max_list)

if __name__ == '__main__':
    exp2()