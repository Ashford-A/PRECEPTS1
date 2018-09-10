
from dryadic.learning.pipelines import PresencePipe
from dryadic.learning.pipelines.transfer import MultiPipe
from dryadic.learning.selection import SelectMeanVar
from dryadic.learning.kbtl.single_domain import SingleDomain
from sklearn.preprocessing import StandardScaler, RobustScaler


class Base(MultiPipe, PresencePipe):

    tune_priors = (
        ('fit__margin', (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)),
        ('fit__sigma_h', (0.04, 0.08, 0.1, 0.12, 0.16, 0.24)),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = SingleDomain(latent_features=5, max_iter=500, stop_tol=0.05)

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Long_iter(Base):

    fit_inst = SingleDomain(latent_features=5, max_iter=1e5, stop_tol=1e-3)


class Features_tune(Base):

    tune_priors = (
        ('fit__margin', (0.75, 1.0, 1.25)),
        ('fit__sigma_h', (1./11, 1./7)),
        ('fit__latent_features', (3, 5, 7, 10, 15, 20)),
        )

    fit_inst = SingleDomain(max_iter=1000, stop_tol=0.01)

