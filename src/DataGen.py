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

def randomZipf(a, maxVal):
    v = int (np.random.zipf(a, 1)[0])
    return v if v <= maxVal else min(int(maxVal), int(np.random.zipf(a, 1)[0]))

def generateData(L, N, maxM, maxC, a):
    """
    Steps:
    - zipf exponent ze
        - For each user u:
            + m = number of locations u visits = Zipf(M, ze) (M: number of elements, ze: exponent)
            + Run m times:
                ++ lid = a location that u has not visited = Zipf(L, ze)  and not visited
                ++ freq = number of visits of u to lid = Zipf(maxC, ze) (M: number of elements, ze: exponent)
    :param L: number of locations
    :param N: number of users
    :param maxM: maximum number of locations that a user can visit
    :param maxC: maximum number of visits that a user can contribute to a location
    :param a: zipf exponent, should be greater than 1
    :param outputFile: output file
    :return:
    """
    locs = defaultdict(Counter)
    for u in xrange(N):
        m = randomZipf(a, maxM) # user u visits m locations
        # m = np.random.randint(1,maxM + 1)
        visited = set() #
        for i in xrange(m):
            while True:
                lid = randomZipf(a, L) # location id
                if lid not in visited:
                    visited.add(lid)
                    freq = randomZipf(a, maxC)
                    # add to dictionary (locs) user u visits location lid in freq times
                    locs[lid][u] = freq
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
            for uidx in xrange(1,len(parts),2):
                counter[int(parts[uidx])] = int(parts[uidx+1])
            locs[lid] = counter
    return locs

