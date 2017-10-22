
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.predict.pipelines import ProteinPipe
import HetMan.predict.regressors as regr


# machine learning pipelines for predicting proteomic levels from -omics data
class ElasticNet(regr.ElasticNet, ProteinPipe):
    pass

class SVRrbf(regr.SVRrbf, ProteinPipe):
    pass

class rForest(regr.rForest, ProteinPipe):
    pass

class kNeigh(regr.KNeighbors, ProteinPipe):
    pass

class GaussProc(regr.GaussianProcess, ProteinPipe):
    pass

class GradBoost(regr.GradientBoosting, ProteinPipe):
    pass

