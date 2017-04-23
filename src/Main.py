__author__ = 'ubriela'
import math
import logging
import time
import sys
import random
import numpy as np
import copy

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


differ = Differential(1000)


def noisyEntropy(count, sens, epsilon):
    """
    Add Laplace noise to Shannon entropy
    :param count: actual count
    :param sens: sensitivity
    :param epsilon: privacy loss
    :return:
    """
    if epsilon < Params.MIN_SENSITIVITY/100:
        return count
    else:
        return count + differ.getNoise(sens, epsilon)

