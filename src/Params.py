__author__ = 'ubriela'

import math
import numpy as np
# # Basic parameters
class Params(object):
    DATASET = "gowallaca"

    maxHeight = 5  # maximum tree height, for kdtrees and quadtrees

    p_sigma = 0.6
    gamma = 0.5
    theta = 10

    def __init__(self, seed):
        self.dataset = ""
        self.resdir = ""
        self.locs = None
        self.f_max = None
        self.f_total = None
        self.users = None
        self.map_locs = None

        self.C = 2
        self.M = 5
        self.K = 50 # only consider locations with at least K users

        self.m = 10 # granularity of equal-size grid cell
        self.eps = 1.0  # epsilon
        self.seed = seed # used in generating noisy counts

        self.Percent = 0.3  # budget allocated for split
        self.geoBudget = 'optimal'  # geometric exponent i/3
        self.splitScheme = 'expo'  # exponential mechanism

    def select_dataset(self):

        if Params.DATASET == "gowallany":
            self.dataset = "../dataset/gowalla_NY.txt"
            self.resdir = "../output/"
            self.x_min = 40.6991117951
            self.y_min = -74.0270912647
            self.x_max = 40.7965600333
            self.y_max = -73.9228093667

        if Params.DATASET == "gowallaca":
            self.dataset = "../dataset/gowalla_CA.txt"
            self.resdir = "../output/"
            self.x_min = 40.6991117951
            self.y_min = -74.0270912647
            self.x_max = 40.7965600333
            self.y_max = -73.9228093667

        if Params.DATASET == "gowallasf":
            self.dataset = "../dataset/gowalla_sf.dat"
            self.resdir = "../output/"
            self.x_min = 37.71127146
            self.y_min = -122.51350164
            self.x_max = 37.83266118
            self.y_max = -122.3627126


    def debug(self):
        print 'number of locs', len(self.locs.keys())
        print 'number of users', len(self.users)
        print 'average number of users per location', np.average([len(self.locs[lid]) for lid in self.locs.keys()])
        print 'average number of locations per user', np.average([len(self.users[uid]) for uid in self.users.keys()])
        print 'maximum number of locations per user', np.max([len(self.users[uid]) for uid in self.users.keys()])
        print 'Maximum number of visits of a user contributes to a location', self.f_max
        print 'Average number of visits of a user to a location', (self.f_total + 0.0)/len(self.locs)