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
    map_locs = {} # key = loc_id, value = (lat, lng)
    locs = {} # key = loc_id, value = {user_id, frequency}
    f_max = 0 # maximum frequency
    f_total = 0 # total frequency
    users = {} # key = user_id, value = {loc_id, frequency}
    min_x, min_y, max_x, max_y = 90, 180, -90, -180
    with open(file, 'r') as ins:
        for row in ins:
            f_total = f_total + 1
            row = row.split()
            user_id = int(row[0])
            loc_id = int(row[4])
            # if int(loc_id) == 32768:
            #     print "xxx"
            if locs.has_key(loc_id):
                if locs[loc_id].has_key(user_id):
                    locs[loc_id][user_id] = locs[loc_id][user_id] + 1
                    f_max = max(f_max, locs[loc_id][user_id])
                else:
                    locs[loc_id][user_id] = 1
            else:
                locs[loc_id] = {user_id : 1}


            if users.has_key(user_id):
                if users[user_id].has_key(loc_id):
                    users[user_id][loc_id] = users[user_id][loc_id] + 1
                else:
                    users[user_id][loc_id] = 1
            else:
                users[user_id] = {loc_id : 1}

            # print row[2]
            # print min_x, row[2]
            min_x = min(min_x, float (row[2]))
            min_y = min(min_y, float (row[3]))

            max_x = max(max_x, float (row[2]))
            max_y = max(max_y, float (row[3]))

            if not map_locs.has_key(loc_id):
                map_locs[loc_id] = (float (row[2]), float (row[3]))

    print min_x, min_y, max_x, max_y
    # print 'f_max, f_total', f_max, f_total
    return locs, f_max, f_total, users, map_locs

# locs = read_checkins("gowalla_HW.txt")
# print 'number of locs', len(locs.keys())

# for key in locs.keys():
    # print key, locs[key].values()