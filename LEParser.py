__author__ = 'ubriela'

import numpy as np
from os import walk
import csv
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from sets import Set

# field_names = ["user_id", "datetime", "latitude", "longitude", "location_id"]
def read_checkins(file):
    d_locs = {} # key = loc_id, value = {user_id, frequency}
    f_max = 0 # maximum frequency
    f_total = 0 # total frequency
    users = Set([]) # a set of user_id
    with open(file, 'r') as ins:
        for row in ins:
            f_total = f_total + 1
            row = row.split()
            loc_id = row[4]
            if d_locs.has_key(loc_id):
                if d_locs[loc_id].has_key(row[0]):
                    d_locs[loc_id][row[0]] = d_locs[loc_id][row[0]] + 1
                    f_max = max(f_max, d_locs[loc_id][row[0]])
                else:
                    d_locs[loc_id][row[0]] = 1
            else:
                d_locs[loc_id] = {row[0]: 1}

            users.add(row[0])

    # print 'f_max, f_total', f_max, f_total
    return d_locs, f_max, f_total, users

# d_locs = read_checkins("gowalla_HW.txt")
# print 'number of locs', len(d_locs.keys())

# for key in d_locs.keys():
    # print key, d_locs[key].values()