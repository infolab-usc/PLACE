
import scipy.stats as stats
from Utils import randomEntropy, temporalUncorrelatedEntropy
import matplotlib.pyplot as plt
import numpy as np

def line_graph(xvals, yvals, markers, legends, xlabel, ylabel):
    for i in range(len(yvals)):
        plt.plot(xvals, yvals[i], markers[i],  color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legends, loc='upper right')
    plt.show()

def distribution_pdf(users):
    """
    distribution of user entropy
    :param users:
    :return:
    """
    randEns = []
    uncorEns = []
    for c in users.itervalues():
        vals = c.values()
        if len(vals) > 1:
            randEns.append(randomEntropy(len(vals)))
            uncorEns.append(temporalUncorrelatedEntropy(vals))

    # randEns = [e for e in randEns if e > 0]
    # uncorEns = [e for e in uncorEns if e > 0]

    # print len(randEns), len(uncorEns)
    # for i in range(len(randEns)):
    #     if randEns[i] != uncorEns[i]:
    #         print randEns[i], uncorEns[i]

    densityRand = stats.gaussian_kde(randEns)
    densityRand.covariance_factor = lambda: 0.5
    densityRand._compute_covariance()
    densityUncor = stats.gaussian_kde(uncorEns)
    densityUncor.covariance_factor = lambda: 0.5
    densityUncor._compute_covariance()


    xs = np.linspace(0, 8, 200)
    plt.plot(xs, densityRand(xs), 'r--', label="random entropy")
    plt.plot(xs, densityUncor(xs), 'bs', label="temporal uncorrelated entropy")
    plt.show()