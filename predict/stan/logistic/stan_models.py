
gauss_model = '''
    data {
        int<lower=1> N;         // number of samples
        int<lower=1> G;         // number of genetic features
        
        matrix[N, G] expr;      // RNA-seq expression values
        int<lower=0, upper=1> mut[N];   // mutation status
        
        real<lower=0> alpha;
    }
    
    parameters {
        real intercept;
        vector[G] gn_wghts;
    }
    
    model {
        intercept ~ normal(0, 1.0);
        gn_wghts ~ normal(0, alpha);
        mut ~ bernoulli_logit(intercept + expr * gn_wghts);
    }
'''


cauchy_model = '''
    data {
        int<lower=1> N;         // number of samples
        int<lower=1> G;         // number of genetic features
        
        matrix[N, G] expr;      // RNA-seq expression values
        int<lower=0, upper=1> mut[N];   // mutation status
        
        real<lower=0> alpha;
    }
    
    parameters {
        real intercept;
        vector[G] gn_wghts;
    }
    
    model {
        intercept ~ normal(0, 1.0);
        gn_wghts ~ cauchy(0, alpha);
        mut ~ bernoulli_logit(intercept + expr * gn_wghts);
    }
'''

