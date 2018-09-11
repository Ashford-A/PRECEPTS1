
stan_ridge = """
// Linear Model with Normal Errors
data {
  int N;                                // train number of observations
  int K;                                // number of columns in the design matrix
  vector[N] y;                          // train response
  matrix [N, K] X;                      // train observations
  int N_test;                           // test number of observations
  matrix [N_test, K] X_test;            // test observations  
  int post_pred;                        // posterior prediction estimates * number of iterations * chains - warmup
  int use_log_lik;                      // log likelihood estimates * number of iterations * chains - warmup
}
parameters {
  real alpha;                           // regression coefficient vector
  vector [K] beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] mu_train;
  vector[N_test] mu_test;
  mu_train = alpha + X * beta;
  mu_test = alpha + X_test * beta;
}
model {
  // priors
  alpha ~ normal(0.0, 10);
  beta ~ normal(0.0, 0.1);
  sigma ~ lognormal(0, 2);
  y ~ normal(mu_train, sigma);              // likelihood
}
generated quantities {

  vector[N_test * post_pred] predicted;         // simulate data from the posterior test
  vector[N * post_pred] predicted_train;        // simulate data from the posterior train
  vector[N * use_log_lik] log_lik_train;        // log-likelihood posterior train
  vector[N_test * use_log_lik] log_lik_test;    // log-likelihood posterior train
  vector[N_test] y_pred_test;
  for (i in 1:N_test)
    y_pred_test[i] = normal_rng(mu_test[i], sigma);
  for (i in 1:num_elements(predicted_train)) {
    predicted_train[i] = normal_rng(mu_train[i], sigma);
  }
  for (i in 1:num_elements(log_lik_train)) {
    log_lik_train[i] = normal_lpdf(y[i] | mu_train[i], sigma);
  }
  for (i in 1:num_elements(predicted)) {
    predicted[i] = normal_rng(mu_test[i], sigma);
  }
  for (i in 1:num_elements(log_lik_test)) {
    log_lik_test[i] = normal_lpdf(y[i] | mu_test[i], sigma);
  }
}
"""


stan_ridge_beta_sampling = """
// Faster Ridge regression
// Directly sample the posterior distribution of beta using the closed-form solution
data {
  // number of observations
  int<lower=0> N;
  int<lower=0> K;
  // Precomputed X'X, X'y
  matrix[K, K] XX;
  matrix[K, K] XX_inv;
  vector[K] Xy;
  // priors on alpha
  real<lower=0.> loc_sigma;
  real<lower=0.> df_lambda;
}
parameters {
  // regression coefficient vector
  vector[K] beta;
  real<lower=0.> lambda;
  real<lower=0.> sigma;
}
transformed parameters {
  vector[K] beta_mean;
  cov_matrix[K] beta_cov;
  {
    matrix[K, K] M = inverse(XX - diag_matrix(rep_vector(lambda, K)));
    beta_mean = M * Xy;
    beta_cov = sigma ^ 2 * quad_form_sym(XX_inv, M);
  }
}
model {
  beta ~ multi_normal(beta_mean, beta_cov);
  sigma ~ exponential(loc_sigma);
  lambda ~ chi_square(df_lambda);
}
generated quantities {
}
"""


stan_lasso = """
// Linear Model with Normal Errors
data {
  int N;                                // train number of observations
  int K;                                // number of columns in the design matrix
  vector[N] y;                          // train response
  matrix [N, K] X;                      // train observations
  int N_test;                           // test number of observations
  matrix [N_test, K] X_test;            // test observations  
  int post_pred;                        // posterior prediction estimates * number of iterations * chains - warmup
  int use_log_lik;                      // log likelihood estimates * number of iterations * chains - warmup
}
parameters {
  real alpha;                           // regression coefficient vector
  vector [K] beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] mu_train;
  vector[N_test] mu_test;
  mu_train = alpha + X * beta;
  mu_test = alpha + X_test * beta;
}
model {
  // priors
  alpha ~ normal(0.0, 10);
  beta ~ double_exponential(0.0, 0.1);
  sigma ~ lognormal(0, 2);
  y ~ normal(mu_train, sigma);              // likelihood
}
generated quantities {

  vector[N_test * post_pred] predicted;         // simulate data from the posterior test
  vector[N * post_pred] predicted_train;        // simulate data from the posterior train
  vector[N * use_log_lik] log_lik_train;        // log-likelihood posterior train
  vector[N_test * use_log_lik] log_lik_test;    // log-likelihood posterior train
  vector[N_test] y_pred_test;
  for (i in 1:N_test)
    y_pred_test[i] = normal_rng(mu_test[i], sigma);
  for (i in 1:num_elements(predicted_train)) {
    predicted_train[i] = normal_rng(mu_train[i], sigma);
  }
  for (i in 1:num_elements(log_lik_train)) {
    log_lik_train[i] = normal_lpdf(y[i] | mu_train[i], sigma);
  }
  for (i in 1:num_elements(predicted)) {
    predicted[i] = normal_rng(mu_test[i], sigma);
  }
  for (i in 1:num_elements(log_lik_test)) {
    log_lik_test[i] = normal_lpdf(y[i] | mu_test[i], sigma);
  }
}
"""

stan_gaussian_process = """
data {
  int<lower=1> D;
  int<lower=1> N;
  int<lower=1> N_test;
  vector[D] X[N];
  vector[N] y1;
  vector[D] X_test[N_test];
}
transformed data {
  int<lower=1> N;
  vector[D] x[N+N_test];
  vector[N+N_test] mu;
  //matrix[N+N_test, N+N_test] Sigma;
  cov_matrix[N+N_test] Sigma;
  N = N + N_test;
  for (n in 1:N) x[n] = X[n];
  for (n in 1:N_test) x[N + n] = X_test[n];
  for (i in 1:N) mu[i] = 0;
  for (i in 1:N)
    for (j in 1:N)
      Sigma[i, j] = exp(-dot_self(x[i] - x[j])) + (i == j ? 0.1 : 0.0);
}
parameters {
  vector[N_test] y2;
}
model {
  vector[N] y;
  for (n in 1:N) y[n] = y1[n];
  for (n in 1:N_test) y[N + n] = y2[n];
  y ~ multi_normal(mu, Sigma);
}
"""


stan_lm_normal = """
// Linear Model with Normal Errors
data {
  // number of observations
  int<lower=0> N;
  // response
  vector[N] y;
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // scale of normal prior on regression intercept
  real<lower=0.> scale_alpha;
  // scale of normal prior on regression coefficients
  vector<lower=0.>[K] scale_beta;
  // expected value of the regression error
  real<lower=0.> rate_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
  real<lower=0.> sigma;
}
transformed parameters {
  vector[N] mu;

  mu = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0., scale_alpha);
  beta ~ normal(0., scale_beta);
  sigma ~ exponential(rate_sigma);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
}
"""


stan_ridge_normal = """
/* Bayesian "Ridge" Regression
Linear regression with normal errors and normal prior on regression coefficients.
*/
data {
  // number of observations
  int<lower=0> N;
  // response
  vector[N] y;
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0.> scale_alpha;
  // prior scale on global df   
  real<lower=0.> df_tau;
  real<lower=0.> scale_tau;
  // prior on regression error distribution
  real<lower=0.> rate_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression coefficient vector on standardized scale
  real alpha_z;
  vector[K] beta_z;
  // scale of regression errors
  real<lower=0.> sigma;
  // global scale
  real<lower=0.> tau;
}
transformed parameters {
  // expected value of the response
  vector[N] mu;
  // coefficients
  real alpha;
  vector[K] beta;

  alpha = scale_alpha * alpha_z;
  beta = tau * sigma * beta_z;
  mu = alpha + X * beta;
}
model {
  // hyperpriors
  tau ~ student_t(df_tau, 0., scale_tau);
  // priors
  sigma ~ exponential(rate_sigma);
  alpha_z ~ normal(0., 1.);
  beta_z ~ normal(0., 1.);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  // shrinkage parameter
  // no local scales so only one value
  real kappa;
  // number of effective coefficients
  real m_eff;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
  {
    real inv_sigma2 = 1. / sigma ^ 2;
    real tau2 = tau ^ 2;
    kappa = 1. / (1. + N * inv_sigma2 * tau2);
  }
  m_eff = K * (1 - kappa);
}
"""


stan_lm_student_t_shrinkage="""
// Linear Model with Normal Errors
data {
  // number of observations
  int<lower=0> N;
  // response
  vector[N] y;
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0.> scale_alpha;
  // prior scale on global
  real<lower=0.> df_tau;
  real<lower=0.> scale_tau;
  // prior on regression error distribution
  real<lower=0.> rate_sigma;
  // degrees of freedom of hyperprior on beta
  real<lower=0.> df_lambda;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
transformed data {
  real half_df_lambda = 0.5 * df_lambda;
}
parameters {
  // regression coefficient vector
  real alpha_z;
  vector[K] beta_z;
  real<lower=0.> sigma;
  // global scale
  real<lower=0.> tau;
  // local variances
  vector<lower=0.>[K] lambda2;
}
transformed parameters {
  vector[N] mu;
  real alpha;
  vector[K] beta;

  alpha = scale_alpha * alpha_z;
  beta = tau * sqrt(lambda2) .* beta_z;
  mu = alpha + X * beta;
}
model {
  // hyperpriors
  lambda2 ~ inv_gamma(half_df_lambda, half_df_lambda);
  tau ~ student_t(df_tau, 0., scale_tau);
  // priors
  alpha_z ~ normal(0., 1.);
  beta_z ~ normal(0., 1.);
  sigma ~ exponential(rate_sigma);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  vector[K] kappa;
  real m_eff;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
  {
    real inv_sigma2 = 1. / sigma ^ 2;
    real tau2 = tau ^ 2;
    kappa = 1. ./ (1. + N * inv_sigma2 * tau2 * lambda2);
  }
  m_eff = K - sum(kappa);
}
"""

stan_lm_laplace = """
/*
  Linear Model with Laplace Prior on Coefficients  (Bayesian Lasso)
*/
data {
  // number of observations
  int<lower=0> N;
  // response
  vector[N] y;
  // number of columns in the design matrix X
  int<lower=0> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha 
  real<lower=0.> scale_alpha;
  // Half-Student-t prior on tau
  real<lower=0.> df_tau;
  real<lower=0.> scale_tau;
  // prior on regression error distribution
  real<lower=0.> rate_sigma;
  // keep responses
  int<lower=0, upper=1> use_y_rep;
  int<lower=0, upper=1> use_log_lik;
}
parameters {
  // regression intercept
  real alpha_z;
  // regression coefficients
  vector[K] beta_z;
  // scale of regression errors
  real<lower=0.> sigma;
  // hyper-parameters of coefficients
  real<lower=0.> tau;
  vector<lower=0.>[K] lambda2;
}
transformed parameters {
  vector[N] mu;
  real alpha;
  vector[K] beta;

  alpha = scale_alpha * alpha_z;
  beta = tau * sqrt(lambda2) .* beta_z;
  mu = alpha + X * beta;
}
model {
  // hyperpriors
  tau ~ student_t(df_tau, 0., scale_tau * sigma);
  lambda2 ~ exponential(0.5);
  // priors
  alpha_z ~ normal(0., 1.);
  beta_z ~ normal(0., 1.);
  sigma ~ exponential(rate_sigma);
  // likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  // simulate data from the posterior
  vector[N * use_y_rep] y_rep;
  // log-likelihood posterior
  vector[N * use_log_lik] log_lik;
  // shrinkage factors
  vector[K] kappa;
  // number of effective coefficients
  real m_eff;
  for (i in 1:num_elements(y_rep)) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
  for (i in 1:num_elements(log_lik)) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
  {
    real inv_sigma2 = 1. / sigma ^ 2;
    real tau2 = tau ^ 2;
    kappa = 1. ./ (1. + N * inv_sigma2 * tau2 * lambda2);
  }
  m_eff = K - sum(kappa);
}
"""