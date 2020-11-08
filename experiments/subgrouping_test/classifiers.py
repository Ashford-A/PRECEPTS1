
from ..utilities.classifiers import Ridge, RidgeMoreTune, SVCrbf, Forests
from dryadic.learning.classifiers import Base, LinearPipe, Kernel, Trees
from dryadic.learning.pipelines.base import PipelineError


import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier


class Elastic(Base, LinearPipe):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-7.25, 1.5, 36))),
        ('fit__l1_ratio', tuple(np.linspace(0.17, 0.83, 23))),
        )
 
    fit_inst = SGDClassifier(loss='log', penalty='elasticnet',
                             max_iter=1000, class_weight='balanced')


# TODO: update these class names to be non-duplicated
class SVCrbf(SVCrbf):

    def get_coef(self):
        return dict()
 

class Forests(Forests):

    def get_coef(self):
        if self.fit_genes is None:
            raise PipelineError("Gene coefficients only available once "
                                "the pipeline has been fit!")

        return {gene: coef for gene, coef in
                zip(self.fit_genes,
                    self.named_steps['fit'].feature_importances_)}

