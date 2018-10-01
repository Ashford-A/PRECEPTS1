
from dryadic.learning.pipelines import TransferPipe, PresencePipe
from dryadic.learning.selection import SelectMeanVar
from dryadic.learning.kbtl.multi_domain import MultiDomain
from sklearn.preprocessing import StandardScaler, RobustScaler


class Base(TransferPipe, PresencePipe):

    tune_priors = (
        ('fit__margin', (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)),
        ('fit__sigma_h', (0.04, 0.08, 0.1, 0.12, 0.16, 0.24)),
        )

    feat_inst = SelectMeanVar(mean_perc=95, var_perc=95)
    norm_inst = StandardScaler()
    fit_inst = MultiDomain(latent_features=2, max_iter=500, stop_tol=0.05)

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class Long_iter(Base):

    fit_inst = MultiDomain(latent_features=5, max_iter=1e5, stop_tol=1e-3)


class Features_tune(Base):

    tune_priors = (
        ('fit__margin', (0.75, 1.0, 1.25)),
        ('fit__sigma_h', (1./11, 1./7)),
        ('fit__latent_features', (3, 4, 5, 6, 7, 8)),
        )

    fit_inst = MultiDomain(max_iter=500, stop_tol=0.05)

