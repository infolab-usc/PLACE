import unittest
import logging

from filter_data import filter_gowalla, filter_yelp
from LEStats import readCheckins, cellStats, entropyStats, otherStats
from plots import distribution_pdf, line_graph, scatter_LE
from LEBounds import globalSensitivy, localSensitivity, precomputeSmoothSensitivity, getSmoothSensitivity
from Params import Params
from Utils import CEps2Str, samplingUsers, transformDict, topKValues
from Metrics import KLDiv, KLDivergence2, typeLE, CatScore
from Main import evalSS, actualEntropy, evalBL, evalGeoI, perturbedLocationEntropy
from Differential import Differential
import sklearn.metrics as metrics
import numpy as np
from DataGen import generateData, writeData, readData

class TestFunctions(unittest.TestCase):
    # @unittest.skip
    def setUp(self):
        # init parameters
        self.log = logging.getLogger("debug.log")
        self.p = Params(1000)
        self.p.select_dataset()

        # self.p.locs, self.p.users, self.p.locDict = readCheckins(self.p)

        self.p.locs = readData("../output/dense.txt")
        self.p.users = transformDict(self.p.locs)

    # @unittest.skip
    def testMain(self):
        # Discretize
        # self.p.locs = cellStats(self.p)
        # self.p.users = transformDict(self.p.locs)
        # distribution_pdf(self.p.locs)

        E_actual = actualEntropy(self.p.locs)

        # Visualization
        le = sorted(list(E_actual.iteritems()), key=lambda x:x[1], reverse=True)    # decrease entropy
        locIds = [t[0] for t in le]
        LEVals = [t[1] for t in le]
        scatter_LE(LEVals, "Location Id", "Entropy")

        E_noisy = perturbedLocationEntropy(self.p, "SS")
        perturbedLEVals = [E_noisy.get(id, Params.DEFAULT_ENTROPY) for id in locIds]
        scatter_LE(perturbedLEVals, "Location Id", "Entropy")

        # evalSS(self.p, E_actual)
        # evalBL(self.p, E_actual)
        # evalGeoI(self.p, E_actual)


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
        C, eps, K = 10, 1.0, 50

        # Baseline sensitivity (max C)
        max_C = 100
        max_gs = globalSensitivy(max_C)
        max_gsy = [max_gs] * len(nx)

        # global sensitivity (limit C)
        gs = globalSensitivy(C)
        gsy = [gs] * len(nx)

        # smooth sensitivity
        ss = getSmoothSensitivity([C], [eps])
        ssy = [v * 2 for v in ss[CEps2Str(C, eps)][:100]]

        # local sensitivity
        K = 20
        ls = localSensitivity(C, K)
        lsy = [ls] * len(nx)
        ny = [max_gsy, gsy, ssy, lsy]
        markers = ["o", "-", "--", "+"]
        legends = ["Global (Max C)", "Global (Limit C)", "Smooth", "Local"]
        line_graph(nx, ny, markers, legends, "Number of users (n)", "Sensitivity")

    @unittest.skip
    def testLEBounds(self):
        eps_list = [0.4, 0.7]
        for eps in eps_list:
            precomputeSmoothSensitivity(eps)

        # print getSmoothSensitivity([1,2,3,4], [0.1])

    @unittest.skip
    def testMetrics(self):
        P = [1,2,3,4,5,6,7,8,9]
        Q = [1,2,4,8,7,6,5,8,9]
        self.assertEqual(True, abs(KLDivergence2(P, Q) - KLDiv(P, Q)) < 1e-6)

        true = [1,2,3,4,5,6,7,8,9]
        predicted = [1,2,3,4,5,6,7,8,9]
        self.assertEqual(1, CatScore(true, predicted))

    @unittest.skip
    def testDifferential(self):
        differ = Differential(1000)
        RTH = (34.020412, -118.289936)
        radius = 500.0  # default unit is meters
        eps = np.log(2)
        for i in range(100):
            (x, y) = differ.getTwoPlanarNoise(radius, eps)
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


    @unittest.skip
    def test_filter_gowalla(self):
        filter_gowalla(self.p)
        filter_yelp(self.p)

    @unittest.skip
    def testDataGen(self):
        SPARSE_N = int(self.p.MAX_N / 10)
        DENSE_N = int(self.p.MAX_N * 10)

        np.random.seed(self.p.seed)
        writeData(generateData(1e+3, SPARSE_N, self.p.MAX_M, self.p.MAX_C, 2), "../output/sparse.txt")
        writeData(generateData(1e+3, DENSE_N, self.p.MAX_M, self.p.MAX_C, 2), "../output/dense.txt")

        readData("../output/sparse.txt")