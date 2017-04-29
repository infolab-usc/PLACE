import numpy as np
from collections import defaultdict, Counter

def writeData(locs, outputFile):
    """
    Write generated data
    for each line: locationId,userid1,visits1,userid3,visits2,.....
    :param locs:
    :param outputFile:
    :return:
    """
    with open(outputFile, "w") as f:
        for lid, c in locs.iteritems():
            line = str(lid)
            for u, freq in c.iteritems():
                line += "," + str(u) + "," + str(freq)
            f.write(line + "\n")
    f.close()

def generateData(L, N, maxM, maxC, a):
    """
    Steps:
    - zipf exponent ze
        - For each user u:
            + m = number of locations u visits = Zipf(M, ze) (M: number of elements, ze: exponent)
            + Run m times:
                ++ l = a location that u has not visited = Zipf(L, ze)  and not visited
                ++ c = number of visits of u to l = Zipf(maxC, ze) (M: number of elements, ze: exponent)
    :param L: number of locations
    :param N: number of users
    :param maxM: maximum number of locations that a user can visit
    :param maxC: maximum number of visits that a user can contribute to a location
    :param a: zipf exponent, should be greater than 1
    :param outputFile: output file
    :return:
    """
    locs = defaultdict(Counter)
    for u in range(N):
        m = int(min(maxM, np.random.zipf(a, 1)[0])) # user u visits m locations
        visited = set()
        for i in range(m):
            if len(visited) >= L:
                break
            while True:
                l = int(min(L, np.random.zipf(a, 1)[0]))
                if l not in visited:
                    visited.add(l)
                    c = int(min(maxC, np.random.zipf(a, 1)[0]))
                    # add to dictionary (locs) user u visits location l in c times
                    locs[l][u] = c
                    break
    return locs

def readData(inputFile):
    """
    Read checkin data from file
    :param inputFile:
    :return:
    """
    locs = defaultdict(Counter)
    with open(inputFile, "r") as f:
        for line in f:
            parts = line.split(",")
            lid = int(parts[0])
            counter = Counter()
            for uidx in range(1,len(parts),2):
                counter[int(parts[uidx])] = int(parts[uidx+1])
            locs[lid] = counter
    return locs

