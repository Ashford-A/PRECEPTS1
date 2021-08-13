
from ..utilities.classifiers import Ridge, RidgeMoreTune, SVCrbf, Forests
from dryadic.learning.scalers import center_scale
from dryadic.learning.classifiers import Base
from sklearn.preprocessing import RobustScaler, Normalizer


class RidgeRobust(Ridge):
    """
    Replaces the z-score normalization in the default `Ridge` classifier with
    a z-score normalization that is less sensitive towards outliers.
    """

    norm_inst = RobustScaler()


class RidgeFlat(Ridge):
    """
    Normalizes expression data sample-wise as well as feature-wise; usually
    we only do the latter.
    """

    norm_inst2 = center_scale

    def __init__(self):
        super(Base, self).__init__([
            ('feat', self.feat_inst), ('norm', self.norm_inst),
            ('norm2', self.norm_inst2), ('fit', self.fit_inst)
            ])

