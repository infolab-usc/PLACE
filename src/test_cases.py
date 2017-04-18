import unittest

from filter_data import filter_gowalla, filter_yelp
from LEStats import readCheckins, cellStats, entropyStats, otherStats, transformDict, distribution_pdf

from Params import Params
class TestFunctions(unittest.TestCase):

    def setUp(self):
        # init parameters
        self.p = Params(1000)
        self.p.select_dataset()

    # def test_filter_gowalla(self):
    #     filter_gowalla(self.p)
    #     filter_yelp(self.p)

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

