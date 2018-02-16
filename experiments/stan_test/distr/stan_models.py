
from ....predict.stan_margins import *


base_model = '''
    data {
        int<lower=1> N;         // number of samples
        int<lower=1> G;         // number of genetic features
        matrix[N, G] expr;      // RNA-seq expression values
        int<lower=0, upper=1> mut[N];   // mutation status
    }

    parameters {
        real alpha;
        vector[G] gn_wghts;
    }

    model {
        alpha ~ normal(0, 1);
        gn_wghts ~ normal(0, 1);
        mut ~ bernoulli_logit(alpha + expr * gn_wghts);
    }
'''

cauchy_model = '''
    data {
        int<lower=1> N;         // number of samples
        int<lower=1> G;         // number of genetic features
        matrix[N, G] expr;      // RNA-seq expression values
        int<lower=0, upper=1> mut[N];   // mutation status
    }

    parameters {
        real alpha;
        vector[G] gn_wghts;
    }

    model {
        alpha ~ cauchy(0, 1);
        gn_wghts ~ cauchy(0, 1);
        mut ~ bernoulli_logit(alpha + expr * gn_wghts);
    }
'''


class LogitOptim(StanOptimizing, LogitStan):
        pass

