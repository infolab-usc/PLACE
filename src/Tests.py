import unittest
import logging

from filter_data import filter_gowalla, filter_yelp
from LEStats import readCheckins, cellStats, entropyStats, otherStats, actualEntropy, actualDiversity, actualLocationCount
from plots import distribution_pdf, line_graph, scatter
from LEBounds import globalSensitivy, localSensitivity, precomputeSmoothSensitivity, getSmoothSensitivity
from Params import Params
from Utils import CEps2Str, samplingUsers, transformDict, topKValues
from Metrics import KLDiv, KLDivergence2, typeLE, DivCatScore
from Main import evalEnt, evalBL, evalCountGeoI, perturbeEntropy, perturbeDiversity, evalDiv, evalCountDiff, perturbeCount, evalDivGeoI
from Differential import Differential
import sklearn.metrics as metrics
import numpy as np
from DataGen import generateData, writeData, readData
from multiprocessing import Pool

class TestFunctions(unittest.TestCase):

    def setUp(self):
        # init parameters
        self.p = Params(1000)
        self.p.select_dataset()

        self.log = logging.getLogger("debug.log")

        # load precomputed smooth sensitivity
        # c_list = range(1, 21)
        # eps_list = [0.1, 0.4, 0.7, 1.0]
        # self.ss = getSmoothSensitivity(c_list, eps_list)
        #
        # if Params.DATASET in ["sparse", "medium", "dense"]: # synthetic
        #     self.p.locs = readData(self.p.dataset)
        # else: # real
        #     self.p.locs, self.p.locDict = readCheckins(self.p)
        # self.p.users = transformDict(self.p.locs)

        # Discretize
        # self.p.locs = cellStats(self.p)
        # self.p.users = transformDict(self.p.locs)
        # distribution_pdf(self.p.locs)

        # self.E_actual = actualEntropy(self.p.locs)      # entropy
        # self.D_actual = actualDiversity(self.p.locs)    # diversity

        # self.C_actual = actualLocationCount(self.p, self.p.locDict) # count


    @unittest.skip
    def testMain(self):

        # Visualization
        # le = sorted(list(self.E_actual.iteritems()), key=lambda x:x[1], reverse=True)    # decrease entropy
        # locIds = [t[0] for t in le]
        # LEVals = [t[1] for t in le]
        # scatter(LEVals, "Location Id", "Entropy")

        # E_noisy = perturbedLocationEntropy(self.p, self.ss, "SS")
        # perturbedLEVals = [E_noisy.get(id, Params.DEFAULT_ENTROPY) for id in locIds]
        # scatter(perturbedLEVals, "Location Id", "Entropy")

        # div = sorted(list(self.D_actual.iteritems()), key=lambda x:x[1], reverse=True)
        # locIds = [t[0] for t in div]
        # divVals = [t[1] for t in div]
        # scatter(divVals, "Location Id", "Diversity")

        # D_noisy = perturbedDiversity(self.p)
        # perturbedDVals = [D_noisy.get(id, Params.DEFAULT_DIVERSITY) for id in locIds]
        # scatter(perturbedDVals, "Location Id", "Diversity")

        # cells = sorted(list(self.C_actual.iteritems()), key=lambda x:x[1], reverse=True)
        # cellIds = [t[0] for t in cells]
        # counts = [t[1] for t in cells]
        # scatter(counts, "Cell Id", "Locations")
        #
        # C_noisy = perturbeCount(self.p)
        # perturbedCounts = [C_noisy.get(id, Params.DEFAULT_FREQUENCY) for id in cellIds]
        # scatter(perturbedCounts, "Cell Id", "Locations")

        evalEnt(self.p, self.E_actual, self.ss)
        evalDiv(self.p, self.D_actual)
        evalBL(self.p, self.E_actual)

        # evalCountDiff(self.p, self.C_actual)
        # evalCountGeoI(self.p, self.C_actual)
        # evalDivGeoI(self.p, self.D_actual)

    @unittest.skip
    def testLEParser(self):
        self.p.locs, self.p.users, self.p.locDict = readCheckins(self.p)
        # distribution_pdf(self.p.locs)
        distribution_pdf(self.p.users)
        self.p.users = samplingUsers(self.p.users, Params.MAX_M)
        distribution_pdf(self.p.users)
        entropyStats(self.p.locs)


        # self.p.maxC, self.p.maxM = otherStats(self.p.locs, self.p.users)

        # discretize
        # cells = cellStats(self.p)
        # entropyStats(cells)
        # self.p.maxC, self.p.maxM = otherStats(cells, transformDict(cells))
        # distribution_pdf(cells)
        # distribution_pdf(transformDict(cells))

    @unittest.skip
    def testLEStats(self):
        nx = range(1,100+1)
        C, eps, K = 2, 1.0, 50

        # Baseline sensitivity (max C)
        max_C = 100
        max_gs = globalSensitivy(max_C)
        max_gsy = [max_gs] * len(nx)

        # global sensitivity (limit C)
        gs = globalSensitivy(C)
        gsy = [gs] * len(nx)

        # smooth sensitivity
        ssy = [v * 2 for v in self.ss[CEps2Str(C, eps)][:100]]

        # local sensitivity
        K = 20
        ls = localSensitivity(C, K)
        lsy = [ls] * len(nx)

        # vary n (all bounds)
        ny = [max_gsy, gsy, ssy, lsy]
        markers = ["o", "-", "--", "+"]
        legends = ["Global (Max C)", "Global (Limit C)", "Smooth", "Local"]
        line_graph(nx, ny, markers, legends, "Number of users (n)", "Sensitivity")

        # vary C
        eps_list = [0.1, 0.4, 0.7, 1.0]
        c_list = range(1, 21)
        n = 100
        ss_list = [[self.ss[CEps2Str(c, eps)][n - 1] for c in c_list] for eps in eps_list]

        markers = ["o", "-", "--", "+", "x"]
        legends = ["Eps=" + str(eps) for eps in eps_list]
        line_graph(c_list, ss_list, markers, legends, "C", "Sensitivity")

        # vary n
        c = 10
        ss_list = [[self.ss[CEps2Str(c, eps)][n - 1] for n in nx] for eps in eps_list]
        line_graph(nx, ss_list, markers, legends, "Number of users (n)", "Sensitivity")

        # vary n & C
        c_list = [1, 10, 20]
        legends = ["C=" + str(c) for c in c_list]
        ss_list = [[self.ss[CEps2Str(c, eps)][n - 1] for n in nx] for c in c_list]
        line_graph(nx, ss_list, markers, legends, "Number of users (n)", "Sensitivity")

    @unittest.skip
    def testLEBounds(self):
        # precompute smooth sensitivity
        eps_list = [1.0]
        pool = Pool(processes=len(eps_list))
        pool.map(precomputeSmoothSensitivity, eps_list)
        pool.join()
        for eps in eps_list:
            precomputeSmoothSensitivity(eps)

    @unittest.skip
    def testMetrics(self):
        P = [1,2,3,4,5,6,7,8,9]
        Q = [1,2,4,8,7,6,5,8,9]
        self.assertEqual(True, abs(KLDivergence2(P, Q) - KLDiv(P, Q)) < 1e-6)

        true = [1,2,3,4,5,6,7,8,9]
        predicted = [1,2,3,4,5,6,7,8,9]
        self.assertEqual(1, DivCatScore(true, predicted))

    @unittest.skip
    def testDifferential(self):
        differ = Differential(1000)
        RTH = (34.020412, -118.289936)
        radius = 500.0  # default unit is meters
        eps = np.log(2)
        for i in range(100):
            (x, y) = differ.getPolarNoise(radius, eps)
            print (str(RTH[0] + x * Params.ONE_KM * 0.001) + ',' + str(RTH[1] + y * Params.ONE_KM*1.2833*0.001))

    @unittest.skip
    def testUtils(self):
        values1 = [1,2,3,4,5,6,7,8,9]
        values2 = [1, 2, 3, 9, 5, 6, 7, 8, 4]
        topVals1, topVals2 = topKValues(3, values1), topKValues(3, values2)
        indices1 = [t[1] for t in topVals1]
        indices2 = [t[1] for t in topVals2]
        self.assertEqual([8,7,6], indices1)
        self.assertEqual([3, 7, 6], indices2)
        self.assertEqual(2.0 / 3, metrics.precision_score(indices1, indices2, average="micro"))


    # @unittest.skip
    def test_filter_gowalla(self):
        filter_gowalla(self.p)
        # filter_yelp(self.p)

    @unittest.skip
    def testDataGen(self):
        SPARSE_N = int(self.p.MAX_N / 10)
        MEDIUM_N = int(self.p.MAX_N)
        DENSE_N = int(self.p.MAX_N * 10)

        np.random.seed(self.p.seed)
        # writeData(generateData(1e+3, SPARSE_N, Params.MAX_M, Params.MAX_C, 2), "../dataset/sparse.txt")
        # writeData(generateData(1e+3, MEDIUM_N, Params.MAX_M, Params.MAX_C, 2), "../dataset/medium.txt")
        writeData(generateData(1e+3, DENSE_N, Params.MAX_M, Params.MAX_C, 2), "../dataset/dense.txt")

        # readData("../dataset/sparse.txt")