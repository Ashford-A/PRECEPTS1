
from ....predict.stan_margins import *
import numpy as np
from scipy import stats


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

margin_model = '''
    data {
        int<lower=1> Nw;         // number of wild-type samples
        int<lower=1> Nm;         // number of mutated samples
        int<lower=1> G;         // number of genetic features

        matrix[Nw, G] expr_w;       // wild-type RNA-seq expression values
        matrix[Nm, G] expr_m;       // mutated RNA-seq expression values

        real wt_distr[2];
        real mut_distr[2];
    }

    parameters {
        real alpha;
        vector[G] gn_wghts;
    }

    model {
        vector[Nw] stat_w;
        vector[Nm] stat_m;

        alpha ~ cauchy(0, 0.5);
        gn_wghts ~ cauchy(0, 0.05);

        stat_w = alpha + expr_w * gn_wghts;
        stat_m = alpha + expr_m * gn_wghts;

        stat_w ~ normal(wt_distr[1], wt_distr[2]);
        stat_m ~ normal(mut_distr[1], mut_distr[2]);
    }
'''


class LogitOptim(StanOptimizing, LogitStan):
        pass


class MarginOptim(StanOptimizing, MarginStan):
        pass

