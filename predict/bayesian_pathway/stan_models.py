
from pystan import stan
import numpy as np


model_code = '''
    data {
        int<lower=1> N;     // number of samples
        int<lower=1> G;     // number of genetic features

        real r[N, G];       // RNA-seq expression values
        real c[N, G];       // copy number GISTIC values
        real p[N, G];       // proteomic measurements

        int <lower=G> P;                // number of pathway interactions
        int <lower=1, upper=G> po[P];   // pathway out-edges
        int <lower=1, upper=G> pi[P];   // pathway in-edges
    }
    
    parameters {
        vector<lower=0, upper=1>[P] wght;   // pathway interaction weights
        vector<lower=0.01, upper=10>[2] wght_prior;

        vector<lower=0, upper=1>[G] comb;   // RNA-CNA combinations
        vector<lower=0.01, upper=10>[2] comb_prior;

        vector<lower=0.1>[G] prec;   // precision of activities
        // vector<lower=0.01, upper=20>[2] prec_prior;

        matrix[N, G] act;                   // inferred gene activities
    }

    transformed parameters{
        matrix[N, G] act_sum;
        matrix[N, G] pred_p;

        for (g in 1:G) {
            for (n in 1:N) {
                act_sum[n, g] = (comb[g] * r[n, g]) + ((1.0 - comb[g]) * c[n, g]);

                pred_p[n, g] = 0;
                for (i in 1:P) {
                    if (pi[i] == g)
                        pred_p[n, g] = pred_p[n, g] + act[n, po[i]] * wght[i];
                }
                // print(pred_p[n, g]);
            }
        }
    }

    model {
        for (i in 1:P) {
            wght[i] ~ beta(wght_prior[1], wght_prior[2]);
        }

        for (g in 1:G) {
            comb[g] ~ beta(comb_prior[1], comb_prior[2]);
        }
        
        for (g in 1:G) {
            // prec[g] ~ gamma(prec_prior[1], prec_prior[2]);
            prec[g] ~ gamma(1.5, 10);
        }
        
        for (g in 1:G) {
            for (n in 1:N) {
                act[n, g] ~ normal(act_sum[n, g], pow(prec[g], -1.0));
                p[n, g] ~ normal(pred_p[n, g], 0.01);
            }
        }

    }'''


model_code_ens = '''
    data {
        int<lower=1> N;                 // number of samples
        int<lower=1> G;                 // number of genetic features
        int<lower=1> R;                 // number of known proteomes
        real<lower=0> prec;             // tuning precision

        real r[N, G];       // RNA-seq expression values
        real c[N, G];       // copy number GISTIC values
        real p[N, G];       // proteomic measurements
        real k[N, R];       // inferred proteomic measurements

        int <lower=G> P;                    // number of pathway interactions
        int <lower=1, upper=(G+R)> po[P];   // pathway out-edges
        int <lower=1, upper=(G+R)> pi[P];   // pathway in-edges
    }
    
    parameters {
        vector<lower=0, upper=1>[P] wght;   // pathway interaction weights
        vector<lower=0.1, upper=8>[2] wght_prior;

        vector<lower=0, upper=1>[G] comb;   // RNA-CNA combinations
        vector<lower=0.1, upper=8>[2] comb_prior;
    }

    transformed parameters{
        matrix[N, G] act_sum;
        matrix[N, G] pred_p;

        for (g in 1:G) {
            for (n in 1:N) {
                act_sum[n, g] = (comb[g] * r[n, g]) + ((1.0 - comb[g]) * c[n, g]);
            }
        }

        for (g in 1:G) {
            for (n in 1:N) {
                pred_p[n, g] = 0;

                for (i in 1:P) {
                    if (pi[i] == g) {
                        
                        if (po[i] <= G) {
                            pred_p[n, g] = pred_p[n, g] + act_sum[n, po[i]] * wght[i];
                        
                        } else {
                            pred_p[n, g] = pred_p[n, g] + k[n, po[i] - G] * wght[i];
                        }
                    }
                }
            }
        }
    }

    model {
        for (i in 1:P) {
            wght[i] ~ beta(wght_prior[1], wght_prior[2]);
        }

        for (g in 1:G) {
            comb[g] ~ beta(comb_prior[1], comb_prior[2]);
        }
        
        for (g in 1:G) {
            for (n in 1:N) {
                p[n, g] ~ normal(pred_p[n, g], prec);
            }
        }

    }'''

