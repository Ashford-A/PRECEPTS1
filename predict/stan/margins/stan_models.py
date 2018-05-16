
gauss_model = '''
    data {
        int<lower=1> Nw;         // number of wild-type samples
        int<lower=1> Nm;         // number of mutated samples
        int<lower=1> G;         // number of genetic features

        matrix[Nw, G] expr_w;       // wild-type RNA-seq expression values
        matrix[Nm, G] expr_m;       // mutated RNA-seq expression values

        real wt_distr[2];       // wild-type label distribution
        real mut_distr[2];      // mutated label distribution
        real alpha;             // feature coefficient regularization
    }

    parameters {
        real intercept;
        vector[G] gn_wghts;
    }

    model {
        vector[Nw] stat_w;
        vector[Nm] stat_m;

        intercept ~ normal(0, 1.0);
        gn_wghts ~ normal(0, alpha);

        stat_w = intercept + expr_w * gn_wghts;
        stat_m = intercept + expr_m * gn_wghts;

        target += (normal_lpdf(stat_w | wt_distr[1], wt_distr[2])
                   * ((Nm * 1.0) / Nw));
        target += (normal_lpdf(stat_m | mut_distr[1], mut_distr[2])
                   * ((Nw * 1.0) / Nm));
    }
'''


cauchy_model = '''
    data {
        int<lower=1> Nw;         // number of wild-type samples
        int<lower=1> Nm;         // number of mutated samples
        int<lower=1> G;         // number of genetic features

        matrix[Nw, G] expr_w;       // wild-type RNA-seq expression values
        matrix[Nm, G] expr_m;       // mutated RNA-seq expression values

        real wt_distr[2];       // wild-type label distribution
        real mut_distr[2];      // mutated label distribution
        real alpha;             // feature coefficient regularization
    }

    parameters {
        real intercept;
        vector[G] gn_wghts;
    }

    model {
        vector[Nw] stat_w;
        vector[Nm] stat_m;

        intercept ~ normal(0, 1.0);
        gn_wghts ~ cauchy(0, alpha);

        stat_w = intercept + expr_w * gn_wghts;
        stat_m = intercept + expr_m * gn_wghts;

        target += (normal_lpdf(stat_w | wt_distr[1], wt_distr[2])
                   * ((Nm * 1.0) / Nw));
        target += (normal_lpdf(stat_m | mut_distr[1], mut_distr[2])
                   * ((Nw * 1.0) / Nm));
    }
'''

