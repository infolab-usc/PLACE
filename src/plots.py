
import scipy.stats as stats
from Utils import randomEntropy, temporalUncorrelatedEntropy
import matplotlib.pyplot as plt

import numpy as np

def line_graph(xvals, yvals, markers, legends, xlabel, ylabel):
    for i in range(len(yvals)):
        plt.plot(xvals, yvals[i], markers[i], color='black', markerfacecolor='none')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legends, loc='upper right')
    plt.show()

def scatter(LEVals, xlabel, ylabel):
    plt.scatter(range(len(LEVals)), LEVals, marker="+")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
        if len(vals) >= 1:
            randEns.append(randomEntropy(len(vals)))
            uncorEns.append(temporalUncorrelatedEntropy(vals))

    densityRand = stats.gaussian_kde(randEns)
    densityRand.covariance_factor = lambda: 0.5
    densityRand._compute_covariance()
    densityUncor = stats.gaussian_kde(uncorEns)
    densityUncor.covariance_factor = lambda: 0.5
    densityUncor._compute_covariance()

    xs = np.linspace(0, 8, 200)
    pltRand, = plt.plot(xs, densityRand(xs), 'r--')
    pltUncor, = plt.plot(xs, densityUncor(xs))
    plt.legend((pltRand, pltUncor), ('Random Entropy', 'Uncorrelated Entropy'))
    plt.xlabel('Entropy')
    plt.ylabel('PDF')
    # plt.show()

    bins = np.arange(0,10.1,0.5)
    histRand, edgesRand = np.histogram(randEns, bins)
    histUncor, edgesUncor = np.histogram(uncorEns, bins)
    fig, ax = plt.subplots()
    bar_width = 0.2
    print edgesRand, edgesUncor
    rectsRand = ax.bar(np.array(edgesRand[:-1]), histRand, bar_width, alpha=0.4, color='r')
    rectsUncor = ax.bar(np.array(edgesUncor[:-1]) + bar_width, histUncor, bar_width, alpha=0.4, color='y')
    ax.set_xticks(np.array(edgesRand[:-1]))
    ax.set_xticklabels([str(t) for t in edgesRand])
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Locations')
    ax.legend((rectsRand, rectsUncor), ('Random Entropy', 'Uncorrelated Entropy'))
    plt.show()