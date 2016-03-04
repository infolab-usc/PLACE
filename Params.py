__author__ = 'ubriela'

import math
# # Basic parameters
class Params(object):
    DATASET = "gowallany"

    def __init__(self, seed):
        self.dataset = ""
        self.resdir = ""
        self.d_locs = None
        self.f_max = None
        self.f_total = None
        self.users = None

        self.K = 100 # only consider locations with at least K users
        self.eps = 1.0  # epsilon
        self.seed = seed # used in generating noisy counts

    def select_dataset(self):
        if Params.DATASET == "gowallany":
            self.dataset = "dataset/gowalla_NY.txt"
            self.resdir = "output/"



    def debug(self):
        print 'number of locs', len(self.d_locs.keys())
        print 'number of users', len(self.users)
        print 'f_max, f_total', self.f_max, self.f_total
