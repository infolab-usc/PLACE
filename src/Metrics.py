from Params import Params
from scipy import stats
import math
import numpy as np
# import editdistance
import collections
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from enum import Enum
import sklearn.metrics as metrics
from Utils import topKValues

def formatRes(f):
    '{:06.4f}'.format(f)


class LEType(Enum):
    Sparse = 1
    Medium = 2
    Dense = 3

def typeLE(le):
    if 0 <= le < 3:
        return LEType.Sparse
    elif 3 <= le < 6:
        return LEType.Medium
    else:
        return LEType.Dense

def CatScore(true, predicted):
    """
    Return precision score
    :param true:
    :param predicted:
    :return:
    """
    trueValues = [typeLE(t).value for t in true]
    predictedValues = [typeLE(p).value for p in predicted]
    return metrics.precision_score(trueValues, predictedValues, average="micro")

def TopK(true, predicted):
    """
    return precision score
    :param true:
    :param predicted:
    :return:
    """
    trueTopKIndices = set([t[1] for t in topKValues(Params.TOP_K, true)])
    predictedTopKIndices = set([t[1] for t in topKValues(Params.TOP_K, predicted)])
    return float(len(trueTopKIndices & predictedTopKIndices)) /len(trueTopKIndices)
    # return metrics.precision_score(trueTopKIndices, predictedTopKIndices, average="micro")


def KLDiv(P, Q):
    """
    Returns the KL divergence, K(P || Q) - the amount of information lost when Q is used to approximate P
    :param P:
    :param Q:
    :return:
    """
    divergence = 0.0
    sump, sumq = float(sum(P)), float(sum(Q))
    probP, probQ = [p/sump for p in P], [q/sumq for q in Q]

    for i in range(len(probP)):
        if probP[i] < Params.PRECISION or probQ[i] < Params.PRECISION: continue
        divergence += probP[i] * math.log(probP[i]/probQ[i], Params.base)
    return divergence

def KLDivergence2(pk, pq):
    return stats.entropy(pk, pq, Params.base)

def MSE(actual, noisy):
    """
    Return mean square error
    :param actual:
    :param noisy:
    :return:
    """
    mean_squared_error(actual, noisy)

def RMSE(actual, noisy):
    """
    Return root mean square error
    :param actual:
    :param noisy:
    :return:
    """
    # print len(actual)
    # print len(noisy)
    tmp = mean_squared_error(actual, noisy)
    # print "Tmp", tmp
    return math.sqrt(float(tmp))

    # noisy_vals = []
    # actual_vals = []
    # print len(actual), actual
    # print len(noisy), noisy
    # for lid in noisy.keys():
    #     noisy_vals.append(noisy.get(lid))
    #     if not actual.has_key(lid):
    #         print lid
    #     actual_vals.append(actual.get(lid))

    # print len(actual_vals), actual_vals
    # print len(noisy_vals), noisy_vals

def MRE(actual, noisy):
    """
    Return mean relative error
    :param actual:
    :param noisy:
    :return:
    """
    if len(actual) != len(noisy): return -1
    absErr = np.abs(np.array(actual) - np.array(noisy))
    idx_nonzero = np.where(np.array(actual) != 0)
    absErr_nonzero = absErr[idx_nonzero]
    true_nonzero = np.array(actual)[idx_nonzero]
    relErr = absErr_nonzero / true_nonzero
    return relErr.mean()

# noisy_vals, actual_vals = [], []
# for lid in noisy.keys():
#     noisy_vals.append(noisy.get(lid))
#     actual_vals.append(actual.get(lid))
