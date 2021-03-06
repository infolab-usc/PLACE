import time
import logging

import numpy as np

from Params import Params
from Kd_pure import Kd_pure
from Kd_true import Kd_true
from Kd_standard import Kd_standard
from Kd_hybrid import Kd_hybrid
from Quad_standard import Quad_standard


class KExp(object):
    def __init__(self, data, query_list):
        self.data = data
        self.query_list = query_list
        logging.debug('Getting true query answers...')
        self.trueRes = np.array([self.getTrue(query) for query in query_list])
        self.selectivity()

    def getTrue(self, query):
        """Get true answer by linear search along each dimension"""
        _data = self.data.copy()
        _ndim = _data.shape[0]
        for dim in range(_ndim):
            if _data.shape[1] == 0:
                break
            idx = np.argsort(_data[dim, :], kind='mergesort')
            _data[:, :] = _data[:, idx]
            x = np.searchsorted(_data[dim, :], query[0, dim], side='left')
            y = np.searchsorted(_data[dim, :], query[1, dim], side='right')
            _data = _data[:, x:y + 1]
        return _data.shape[1]

    def selectivity(self):
        sel_array = np.sort(self.trueRes / float(Params.NDATA))
        logging.debug('selectivity min %.2f' % np.min(sel_array))
        logging.debug('selectivity max %.2f' % np.max(sel_array))
        logging.debug('selectivity avg %.2f' % np.average(sel_array))
        logging.debug('selectivity median %.2f' % sel_array[Params.nQuery / 2])
        logging.debug('selectivity first quartile %.2f' % sel_array[Params.nQuery / 4])
        logging.debug('selectivity third quartile %.2f' % sel_array[3 * Params.nQuery / 4])

    def run_Kd_pure(self, param):
        logging.debug('building Kd_pure...')
        tree = Kd_pure(self.data, param)
        start = time.clock()
        tree.buildIndex()
        end = time.clock()
        logging.info('[T] Kd-pure building time: %.2f' % (end - start))
        return self.query(tree)

    def run_Kd_true(self, param):
        logging.debug('building Kd_true...')
        tree = Kd_true(self.data, param)
        start = time.clock()
        tree.buildIndex()
        tree.adjustConsistency()
        end = time.clock()
        logging.info('[T] Kd-true building time: %.2f' % (end - start))
        return self.query(tree)

    def run_Kd_standard(self, param):
        logging.debug('building Kd_standard...')
        tree = Kd_standard(self.data, param)
        start = time.clock()
        tree.buildIndex()
        tree.adjustConsistency()
        end = time.clock()
        logging.info('[T] Kd-standard building time: %.2f' % (end - start))
        return self.query(tree)

    def run_Kd_hybrid(self, param):
        logging.debug('building Kd_hybrid...')
        tree = Kd_hybrid(self.data, param)
        start = time.clock()
        tree.buildIndex()
        tree.adjustConsistency()
        end = time.clock()
        logging.info('[T] Kd-hybrid building time: %.2f' % (end - start))
        return self.query(tree)

    def run_Kd_noisymean(self, param):
        logging.debug('building Kd_noisymean...')
        param.splitScheme = 'noisyMean'
        tree = Kd_standard(self.data, param)
        start = time.clock()
        tree.buildIndex()
        tree.adjustConsistency()
        end = time.clock()
        logging.info('[T] Kd-noisymean building time: %.2f' % (end - start))
        return self.query(tree)


    def run_Quad_baseline(self, param):
        logging.debug('building Quad_baseline...')
        param.geoBudget = 'none'
        tree = Quad_standard(self.data, param)
        tree.buildIndex()
        return self.query(tree)

    def run_Quad_geo(self, param):
        logging.debug('building Quad_geo...')
        param.geoBudget = 'optimal'
        tree = Quad_standard(self.data, param)
        tree.buildIndex()
        return self.query(tree)


        # def run_Quad_geo_vldb(self, param):

    # logging.debug('building Quad_geo_vldb...')
    # param.geoBudget = 'optimal'
    # tree = Quad_standard(self.data, param)
    # tree.buildIndex()
    # tree.adjustConsistency()
    # return self.query(tree)
    #
    # def run_Quad_post_vldb(self, param):
    #        logging.debug('building Quad_post_vldb...')
    #        param.geoBudget = 'none'
    #        tree = Quad_standard_full(self.data, param)
    #        tree.buildIndex()
    #        tree.adjustConsistency()
    #        tree.pruning()
    #        return self.query(tree)

    def query(self, tree):
        """ wrapper for query answering and computing query error """
        #        i_whole, l_whole, l_part = 0, 0, 0
        #        result = []
        #        for query in self.query_list:
        #            res, iw, lw, lp = tree.rangeCount(query)
        #            result.append(res)
        #            i_whole += iw
        #            l_whole += lw
        #            l_part += lp
        #        Res = np.array(result)
        #        print 'i_whole:', float(i_whole)/len(self.query_list)
        #        print 'l_whole:', float(l_whole)/len(self.query_list)
        #        print 'l_part:', float(l_part)/len(self.query_list)
        result = []
        for query in self.query_list:
            result.append(tree.rangeCount(query))
        Res = np.array(result)
        return self.computeError(Res)

    def computeError(self, Res):
        """ Compute median absolute and relative errors """
        absErr = np.abs(Res - self.trueRes)
        idx_nonzero = np.where(self.trueRes != 0)
        absErr_nonzero = absErr[idx_nonzero]
        true_nonzero = self.trueRes[idx_nonzero]
        relErr = absErr_nonzero / true_nonzero
        absErr = np.sort(absErr)
        relErr = np.sort(relErr)
        n_abs = len(absErr)
        n_rel = len(relErr)
        return absErr[int(n_abs / 2)], relErr[int(n_rel / 2)]
