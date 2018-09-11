from sklearn.model_selection import ShuffleSplit,GridSearchCV

import numpy as np
import pystan
from sklearn.base import BaseEstimator, RegressorMixin


class StanModel_(pystan.StanModel):
    def __del__(self):
        """
        This method is being used carelessly in sklearn's GridSearchCV class,
        creating and destroying copies of the estimator, which is causing the
        directory containing the compiled Stan code to be deleted.  It is
        replaced here with an empty method to avoid this problem.  This means
        a potential proliferation of temporary directories.
        """
        pass


class StanOptimizingEstimator(BaseEstimator, RegressorMixin):
    """
    A new sklearn estimator class derived for use with pystan.
    """
    def __init__(self, **kwargs):
        self.data = None
        self.fit_obj = None
        self.summary = None
        self.model_name = None
        self.stan_model = None
        for key, value in kwargs.items():
            setattr(self, key, value)


    def set_model(self, code):
        """
        Sets and compiles a Stan model for this estimator.
        """
        self.model = StanModel_(model_code = code)

    def set_data(self, **kwargs):
        """
        Sets the data for use with this estimator.
        Uses the 'data' keyword argument if provided, else it
        uses the 'make_data' method.
        """
        raise NotImplementedError("Stan predictors must implement their "
                                  "own <set_data> method!")


    def make_data(self, *args, **kwargs):
        """
        A model-specific method for constructing the data to be used by
        the model.  May be limited to the data passed to the Stan model's
        fitter, or may also include other items as well.  Should return
        a dictionary."""
        raise NotImplementedError("Stan predictors must implement their "
                                  "own <make_data> method!")

    def optimize(self, **kwargs):
        """
        Optimizes the estimator based on covariates X and observations y.
        """
        if self.data is None:
            raise ValueError("The data for the experiment has not been set!")
        self.fit_obj = self.model.optimizing(data = self.data, **kwargs)

    def get_params(self, deep = False):
        """
        Gets model parameters.  These are just attributes of the estimator
        as set in __init__ and possibly in other methods.
        """
        return self.__dict__

    def fit(self):
        """
        Fits the estimator based on covariates X and observations y.
        """
        self.optimize()
        for key, value in self.fit_obj.items():
            setattr(self, key, value)

    def transform(self, X, y = None, **fit_params):
        """
        Performs a transform step on the covariates after fitting.
        In the basic form here it just returns the covariates.
        """
        return X

    def predict_(self, X, i):
        """
        Generates a prediction for one sample, based on X, the array of
        covariates and i, a point in that array (1D), or row (2D), etc.
        This must be implemented for each model.
        """
        raise NotImplementedError("")


    @classmethod
    def get_posterior_mean(cls, fit_obj):
        """
        Implemented because get_posterior_mean is (was?) broken in pystan:
        https://github.com/stan-dev/pystan/issues/107
        """
        means = {}
        x = fit_obj.extract()
        for key, value in x.items()[:-1]:
            means[key] = value.mean(axis = 0)
        return means


class Stanlm(StanOptimizingEstimator):

    def set_data(self, data_dict):
        """
        Sets the data for use with this estimator.
        Uses the 'data' keyword argument if provided, else it
        uses the 'make_data' method.
        """

        self.data = data_dict
        for key, value in data_dict.items():
            setattr(self, key, value)


    def make_data(self, **kwargs):
        # Implement a make_data method for the estimator.
        # This tells the sklearn estimator what things to pass along
        # as data to the Stan model.

        if self.data is None:
            raise ValueError("The data for the experiment has not been set!")

        return self.data

    def predict(self):
        """
        Generates a prediction based on X, the array of covariates.
        """
        if self.fit_obj is None:
            raise ValueError("Model has not been fit!")

        prediction = self.fit_obj['predicted']

        return prediction

    def score(self,**kwargs):
        """
        Generates a score for the prediction based on X, the array of
        covariates, and y, the observation.

        Args:
            **kwargs:
        """
        from sklearn.metrics import r2_score

        return r2_score(self.data['y_test'],  self.predict(), sample_weight=None, multioutput='variance_weighted')
