
from pystan import stan
import numpy as np


model_code = '''
    data {
        int<lower=1> N;     // number of samples
        int<lower=1> G;     // number of genetic features

        real r[N, G];       // observed RNA-seq expression values
        real c[N, G];       // observed copy number GISTIC values
        real p[N, G];       // observed proteomic measurements

        int <lower=G> P;            // number of known pathway interactions
        int <lower=1, upper=G> po[P];   // indices of genes left by edges
        int <lower=1, upper=G> pi[P];   // indices of genes entered by edges
    }
    
    parameters {
        // the weights given to RNA-seq expression when determining
        // transcription levels of each gene, and the prior governing the
        // distribution of these weights
        vector<lower=0, upper=1>[G] tx_wght;
        vector<lower=0.1, upper=5>[2] tx_wght_prior;

        // the inferred accuracy of transcription levels in measuring
        // activity for each gene, and the prior governing the distribution
        // of these accuracies
        vector<lower=0.1>[G] tx_acc;
        vector<lower=0.01, upper=20>[2] tx_acc_prior;

        // inferred activity of each gene in each sample
        matrix[N, G] act;

        // the weights given to pathway edges between genes that interact
        // with one another, and the prior governing the distribution of
        // these weights
        vector<lower=0, upper=1>[P] edge_wght;
        vector<lower=0.1, upper=5>[2] edge_wght_prior;
    }

    transformed parameters{
        matrix[N, G] tx;        // inferred transcription levels of genes
        matrix[N, G] pred_p;    // predicted protein levels of genes

        // calculate the transcription level of each gene in each sample using
        // the corresponding observed expression and copy number levels
        for (g in 1:G) {
            for (n in 1:N) {
                pred_p[n, g] = 0;
                tx[n, g] = (tx_wght[g] * r[n, g]) + ((1.0 - tx_wght[g]) * c[n, g]);
            }
        }

        // calculate the predicted protein levels of each gene in each sample
        // given inferred activity level of the gene itself and of the genes
        // that have pathway edges going into the gene
        for (i in 1:P) {
            for (n in 1:N) {
                pred_p[n, pi[i]] = pred_p[n, pi[i]] + act[n, po[i]] * edge_wght[i];
            }
        }
    }

    model {
        // the weights of expression levels in calculating transcription
        // levels follow a distribution that is to be inferred
        for (g in 1:G) {
            tx_wght[g] ~ beta(tx_wght_prior[1], tx_wght_prior[2]);
        }

        // accuracies of transcription levels in measuring gene activity
        // levels follow a distribution that is to be inferred
        for (g in 1:G) {
            tx_acc[g] ~ gamma(tx_acc_prior[1], tx_acc_prior[2]);
        }
        
        // pathway edge weights follow a distribution that is to be inferred
        for (i in 1:P) {
            edge_wght[i] ~ beta(edge_wght_prior[1], edge_wght_prior[2]);
        }

        // gene activity levels are transcription levels with noise added,
        // observed protein levels are predicted protein levels plus noise
        for (g in 1:G) {
            for (n in 1:N) {
                act[n, g] ~ normal(tx[n, g], pow(tx_acc[g], -1.0));
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

        vector<lower=0, upper=1>[G] tx_wght;   // RNA-CNA tx_wghtinations
        vector<lower=0.1, upper=8>[2] tx_wght_prior;
    }

    transformed parameters{
        matrix[N, G] tx;
        matrix[N, G] pred_p;

        for (g in 1:G) {
            for (n in 1:N) {
                tx[n, g] = (tx_wght[g] * r[n, g]) + ((1.0 - tx_wght[g]) * c[n, g]);
            }
        }

        for (g in 1:G) {
            for (n in 1:N) {
                pred_p[n, g] = 0;

                for (i in 1:P) {
                    if (pi[i] == g) {
                        
                        if (po[i] <= G) {
                            pred_p[n, g] = pred_p[n, g] + tx[n, po[i]] * wght[i];
                        
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
            tx_wght[g] ~ beta(tx_wght_prior[1], tx_wght_prior[2]);
        }
        
        for (g in 1:G) {
            for (n in 1:N) {
                p[n, g] ~ normal(pred_p[n, g], prec);
            }
        }

    }'''

