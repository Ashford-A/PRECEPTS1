
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.selection import SelectMeanVar

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Base(PresencePipe):
    """Support Vector Classifiers with various kernels. 

    The `Base` model corresponds to the simplest linear kernel; other choices
    of kernels are implemented below.

    Note that a unique tuning grid for the `C` regularization parameter needs
    to be specified in each version of this model due to the differences in
    characteristics associated with each kernel.
    """

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-6.3, -2.8, 36))),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = SVC(kernel='linear', probability=True,
                   cache_size=500, class_weight='balanced')

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Kernel_quad(Base):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-1, 5, 9))),
        ('fit__gamma', tuple(10 ** np.array([-8.5, -7, -5.5, -4]))),
        )
 
    fit_inst = SVC(kernel='poly', degree=2, coef0=1, probability=True,
                   cache_size=500, class_weight='balanced')


class Kernel_cubic(Base):

    tune_priors = (
        ('fit__C', (1e-9, 1e-6, 1e-4, 1e-2, 1e0, 1e3)),
        ('fit__gamma', (1e-6, 1e-3, 1e-2, 1e1)),
        )
 
    fit_inst = SVC(kernel='poly', degree=3, coef0=1, probability=True,
                   cache_size=500, class_weight='balanced')


class Big_cache(Base):

    tune_priors = (
        ('fit__C', (1e-9, 1e-6, 1e-4, 1e-2, 1e0, 1e3)),
        ('fit__gamma', (1e-6, 1e-3, 1e-2, 1e1)),
        )
 
    fit_inst = SVC(kernel='poly', degree=3, coef0=1, probability=True,
                   cache_size=2000, class_weight='balanced')


class Kernel_poly(Base):

    tune_priors = (
        ('fit__degree', (2, 3, 4, 5)),
        ('fit__coef0', (0, 2, 5)),
        ('fit__C', (1, 10, 100)),
        )
 
    fit_inst = SVC(kernel='poly', gamma=1e-6, probability=True,
                   cache_size=500, class_weight='balanced')


class Kernel_radial(Base):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-1.05, 5, 12))),
        ('fit__gamma', (1e-9, 1e-7, 1e-5)),
        )
 
    fit_inst = SVC(kernel='rbf', probability=True,
                   cache_size=500, class_weight='balanced')


class Radial_fixed_gamma(Base):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-0.3, 3.2, 36))),
        )
 
    fit_inst = SVC(kernel='rbf', probability=True, gamma=1e-6,
                   cache_size=500, class_weight='balanced')


class Radial_meanvar(Base):

    tune_priors = (
        ('feat__mean_perc', (40, 55, 70)),
        ('feat__var_perc', (40, 55, 70)),
        ('fit__C', (1, 5, 10, 100)),
        )

    fit_inst = SVC(kernel='rbf', probability=True, gamma=1e-6,
                   cache_size=500, class_weight='balanced')

