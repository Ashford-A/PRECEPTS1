
from ....predict.pipelines import PresencePipe
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Base(PresencePipe):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-7, -1.25, 24))),
        )

    norm_inst = StandardScaler()
    fit_inst = SVC(kernel='linear', probability=True,
                   cache_size=500, class_weight='balanced')

    def __init__(self):
        super().__init__([('norm', self.norm_inst), ('fit', self.fit_inst)])


class Kernel_quad(Base):

    tune_priors = (
        ('fit__C', (5e-7, 1e-5, 1e-3, 1e-2, 1e0, 5e2)),
        ('fit__gamma', (1e-7, 1e-4, 1e-3, 1e0)),
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
        ('fit__degree', (2, 3, 4)),
        ('fit__coef0', (-2, 0, 2, 5)),
        ('fit__C', (1e-1, 1e1)),
        )
 
    fit_inst = SVC(kernel='poly', gamma=1e-3, probability=True,
                   cache_size=500, class_weight='balanced')


class Kernel_radial(Base):

    tune_priors = (
        ('fit__C', (1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6)),
        ('fit__gamma', (1e-9, 1e-5, 1e-2, 1e2)),
        )
 
    fit_inst = SVC(kernel='rbf', probability=True,
                   cache_size=500, class_weight='balanced')

