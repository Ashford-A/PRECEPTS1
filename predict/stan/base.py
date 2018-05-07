
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import pystan


class BaseStan(BaseEstimator):

    model_name = "BasePredict"

    def __init__(self, model_code):
        self.model_code = model_code

        self.data_dict = None
        self.stan_model = None
        self.fit_obj = None

    def get_data_dict(self, omic, pheno, **fit_params):
        raise NotImplementedError("Stan predictors must implement their "
                                  "own <get_data_dict> method!")

    def run_model(self, **fit_params):
        raise NotImplementedError("Stan predictors must implement their "
                                  "own <get_data_dict> method!")

    def get_var_means(self):
        raise NotImplementedError("Stan predictors must implement their "
                                  "own <get_var_means> method!")

    def fit(self, X, y=None, **fit_params):
        if 'verbose' not in fit_params:
            fit_params['verbose'] = False

        self.stan_model = pystan.StanModel(model_code=self.model_code,
                                           model_name=self.model_name,
                                           verbose=fit_params['verbose'])

        self.data_dict = self.get_data_dict(omic=X, pheno=y, **fit_params)
        self.run_model(**fit_params)

        return self


class StanClassifier(BaseStan, ClassifierMixin):

    def calc_pred_labels(self, omic):
        raise NotImplementedError("Stan predictors must implement their "
                                  "own <calc_pred_labels> method!")

    def calc_pred_p(self, pred_labels):
        raise NotImplementedError("Stan predictors must implement their "
                                  "own <calc_pred_p> method!")

    def predict_proba(self, X):
        if self.fit_obj is None:
            raise NotFittedError("Stan classifier has not been fit yet!")

        return self.calc_pred_p(self.calc_pred_labels(X))


class StanOptimizing(BaseStan):

    def get_var_means(self):
        var_means = {}

        for var, vals in self.fit_obj.items():
            if vals.shape:
                var_means[var] = vals
            else:
                var_means[var] = float(vals)

        return var_means

    def run_model(self, **fit_params):
        if 'iter' not in fit_params:
            fit_params['iter'] = 1e4

        self.fit_obj = self.stan_model.optimizing(data=self.data_dict,
                                                  **fit_params)


class StanVariational(BaseStan):

    def get_var_means(self):
        var_means = {}

        for var, val in zip(self.fit_obj['sampler_param_names'],
                            self.fit_obj['mean_pars']):
            var_parse = var.split('.')

            if len(var_parse) == 1:
                var_means[var_parse[0]] = val

            else:
                if var_parse[0] in var_means:
                    var_means[var_parse[0]] += [val]
                else:
                    var_means[var_parse[0]] = [val]

        return var_means

    def run_model(self, **fit_params):
        self.fit_obj = self.stan_model.vb(data=self.data_dict, **fit_params)


class StanSampling(BaseStan):

    def __init__(self, model_code):
        self.fit_summary = None
        super().__init__(model_code)

    def get_var_means(self):
        var_means = {}

        for var, val in zip(self.fit_obj.flatnames, self.fit_summary[:, 0]):
            var_parse = var.split('[')

            if len(var_parse) == 1:
                var_means[var_parse[0]] = val

            else:
                if var_parse[0] in var_means:
                    var_means[var_parse[0]] += [val]
                else:
                    var_means[var_parse[0]] = [val]

        return var_means

    def run_model(self, **fit_params):
        if 'chains' not in fit_params:
            fit_params['chains'] = 1

        if 'iter' not in fit_params:
            fit_params['iter'] = 50

        self.fit_obj = self.stan_model.sampling(data=self.data_dict,
                                                **fit_params)

        if fit_params['verbose']:
            print("Fitting has finished!")

        self.fit_summary = dict(self.fit_obj.summary())['summary']

