import math
import sys
import numpy as np
from collections import defaultdict
from Params import Params
from Utils import getSmoothSensitivityFile
from Utils import CEps2Str

def diversitySensitivity(M):
    return M * math.log(2, Params.base) # add or remove one users change diversity by maximum ln(2)

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

def smoothSensitivity(C, n, eps, delta, earlyTermination=True):
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
    maxSS = 0
    for k in xrange(int(Params.MAX_N)):
        left_ls, right_ls = localSensitivity(C, max(1, n - k)), localSensitivity(C, n + k)
        maxSS = max(maxSS, math.exp(-k*beta) * max(left_ls, right_ls))
        stopCond1 = math.exp(-k * beta) * gs < maxSS # current smooth sensitivy reaches maximum
        stopCond2 = n + k > (C / (math.log(C, Params.base) - 1) + 1)
        if earlyTermination and stopCond1 and stopCond2: break
    return maxSS

def precomputeSmoothSensitivity(eps):
    """
    Precompute smooth sensitivity given epsilon
    :param eps:
    :return:
    """
    for C in range(1, Params.MAX_C_SS + 1):
        outputFile = getSmoothSensitivityFile(C, eps)
        with open(outputFile, "w") as f:
            lines = ""
            for n in xrange(1, int(Params.MAX_N)):
                ss = smoothSensitivity(C, n, eps, Params.DELTA)
                if n > (C / (math.log(C, Params.base) - 1) + 1) and ss < Params.MIN_SENSITIVITY: break  # stop condition 2
                lines += str(n) + "\t" + str(ss) + "\n"
            f.write(lines)
        f.close()

# precomputeSmoothSensitivity(1.0)

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
            inputFile = getSmoothSensitivityFile(C, eps)
            data = np.loadtxt(inputFile, dtype=float, delimiter="\t")
            dict[CEps2Str(C, eps)] = [v for v in data[:,1]]
    return dict