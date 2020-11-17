
from dryadic.learning.pipelines.base import OmicPipe
from dryadic.learning.selection import SelectMeanVar

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


class Base(OmicPipe):
    """An abstract class for the set of standard transformers.

    The transformers in this module are designed to use all available -omic
    features save those with very low expression in order to get the fullest
    possible picture of the features that can be used to cluster a given task.

    """

    feat_inst = SelectMeanVar(mean_perc=90, var_perc=100)
    norm_inst = StandardScaler()

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


class OmicPCA(Base):
    fit_inst = PCA()


class OmicTSNE(Base):
    fit_inst = TSNE()


class OmicUMAP(Base):
    fit_inst = UMAP()

