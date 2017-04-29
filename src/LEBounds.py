import math
import sys
import numpy as np
from collections import defaultdict
from Params import Params
from Utils import getSmoothSensitivityFile
from Utils import CEps2Str

def frequencySensitivity(C, M):
    """
    Add/remove one user affects M locations, each location affects up to C checkins
    :param C:
    :param M:
    :return:
    """
    return C*M

def localSensitivity(C, n):
    """
    Local sensitivity depends on C and the number of users visiting the location n
    :param C:
    :param n:
    :return:
    """
    if n == 1:
        return math.log(2, Params.base)
    elif C == 1:
        return math.log((n+1.0)/n, Params.base)
    else:
        part1 = math.log((n - 1.0) / (n - 1.0 + C), Params.base) + C / (n - 1.0 + C) * math.log(C, Params.base)
        part2 = math.log((n + 0.0) / (n + C), Params.base) + (C + 0.0) / (n + C) * math.log(C, Params.base)
        part3 = math.log(1 + 1.0 / (math.exp(math.log(n + 1, Params.base) - math.log(C, Params.base) / (C - 1.0) + math.log(math.log(C, Params.base) / (C - 1.0), Params.base)) + 1))
        return max(part1, part2, part3)

def globalSensitivy(C):
    """
    Tight bound for the global sensitivity of LE.
    :param C: is the maximum visits a user contributes to a location
    :return: global sensitivity
    """
    worstCaseDiversity = math.log(2, Params.base) # there are only two users
    worstCaseFrequency = math.log(C, Params.base) -  math.log(math.log(C, Params.base), Params.base) - 1 if C > 1 else 0
    return max(worstCaseDiversity, worstCaseFrequency)

def smoothSensitivity(C, n, eps, delta):
    """
    Smooth sensitity upper bound the local sensitivity.
    :param n:
    :param C:
    :param M:
    :param eps:
    :param delta
    :return:
    """
    gs = globalSensitivy(C)
    beta = eps / (2.0 * math.log(2.0 / delta, math.e))
    stopCond1, stopCond2 = False, False
    maxSS = 0
    for k in range(Params.MAX_N):
        currSS = 0
        if not stopCond1:
            ls = localSensitivity(C,  max(1, n - k))
            currSS = max(currSS, math.exp(-k*beta) * ls)
            stopCond1 = math.exp(-k*beta) * gs < maxSS
        if not stopCond2:
            ls = localSensitivity(C, n + k)
            currSS = max(currSS, math.exp(-k*beta) * ls)
            stopCond2 = n + k > (C / (math.log(C, Params.base) - 1) + 1)
        maxSS = max(maxSS, currSS)
        if stopCond1 and stopCond1: break

    return maxSS

def precomputeSmoothSensitivity(eps):
    """
    Precompute smooth sensitivity given epsilon
    :param eps:
    :return:
    """
    print eps, Params.MAX_C_SS
    for C in range(1, Params.MAX_C_SS + 1):
        print "precomputeSmoothSensitivity ", C
        outputFile = getSmoothSensitivityFile(C, eps)
        with open(outputFile, "a") as f:
            lines = ""
            for n in range(1, Params.MAX_N):
                ss = smoothSensitivity(C, n, eps, Params.DELTA)
                if ss < Params.MIN_SENSITIVITY: break
                lines += str(n) + "\t" + str(ss) + "\n"
            f.write(lines)
        f.close()

def getSmoothSensitivity(C_list, eps_list):
    """
    Get precomputed smooth sensitivity
    :param C_list:
    :param eps_lis:
    :return: mapping from C and eps to sensitivity list
    """
    dict = defaultdict(list)
    for C in C_list:
        for eps in eps_list:
            key = CEps2Str(C, eps)
            inputFile = getSmoothSensitivityFile(C, eps)
            data = np.loadtxt(inputFile, dtype=float, delimiter="\t")
            value = [v for v in data[:,1]]
            dict[key] = value
    return dict