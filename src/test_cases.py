import unittest

from filter_data import filter_gowalla, filter_yelp
from LEStats import readCheckins, cellStats, entropyStats, otherStats, transformDict
from plots import distribution_pdf, line_graph
from LEBounds import globalSensitivy, localSensitivity, precomputeSmoothSensitivity, getSmoothSensitivity
from Params import Params
from Utils import CEps2Str
from Metrics import KLDivergence, KLDivergence2

class TestFunctions(unittest.TestCase):

    def setUp(self):
        # init parameters
        self.p = Params(1000)
        self.p.select_dataset()

    @unittest.skip
    def test_filter_gowalla(self):
        filter_gowalla(self.p)
        filter_yelp(self.p)

    @unittest.skip
    def testLEParser(self):
        self.p.locs, self.p.users, self.p.locDict = readCheckins(self.p)
        distribution_pdf(self.p.locs)
        distribution_pdf(self.p.users)
        # entropyStats(self.p.locs)
        # self.p.maxC, self.p.maxM = otherStats(self.p.locs, self.p.users)

        # discretize
        cells = cellStats(self.p)
        # entropyStats(cells)
        # self.p.maxC, self.p.maxM = otherStats(cells, transformDict(cells))
        distribution_pdf(cells)
        distribution_pdf(transformDict(cells))

    # @unittest.skip
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
        eps_list = [0.1, 0.5, 1.0, 5.0, 10.0]
        for eps in eps_list:
            precomputeSmoothSensitivity(eps)

        print getSmoothSensitivity([1,2,3,4], [0.1])

    @unittest.skip
    def testMetrics(self):
        P = [1,2,3,4,5,6,7,8,9]
        Q = [1,2,4,8,7,6,5,8,9]
        self.assertEqual(True, abs(KLDivergence2(P, Q) - KLDivergence(P, Q)) < 1e-6)