
functions {

  // function to obtain change point matrix for all times 
  
  matrix get_changepoint_matrix(vector t, vector t_change, int T, int S) {
    // Assumes t and t_change are sorted.
    matrix[T, S] A;
    row_vector[S] a_row;
    int cp_idx;

    // Start with an empty matrix.
    A = rep_matrix(0, T, S);
    a_row = rep_row_vector(0, S);
    cp_idx = 1;

    // Fill in each row of A.
    for (i in 1:T) {
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;
        cp_idx = cp_idx + 1;
      }
      A[i] = a_row;
    }
    return A;
  }

  // Logistic trend functions

  // function to compute the adjustment factors gamma
  
  vector logistic_gamma(real k, real m, vector delta, vector t_change, int S) {
    vector[S] gamma;     // vector for adjusted offsets, for piecewise continuity
    vector[S + 1] k_s;   // vector for actual rate in each segment
    real m_pr;

    // Compute the rate in each segment
    k_s = append_row(k, k + cumulative_sum(delta));

    // Piecewise offsets (adjustment for countinuity of growth trend)
    m_pr = m; // The offset in the previous segment
    for (i in 1:S) {
      gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
      m_pr = m_pr + gamma[i];  // update for the next segment
    }
    return gamma;
  }

  // function to compute the logistic trend
  
  vector logistic_trend(
    real k,
    real m,
    vector delta,
    vector t,
    vector cap,
    matrix A,
    vector t_change,
    int S
  ) {
    vector[S] gamma;

    gamma = logistic_gamma(k, m, delta, t_change, S);
    return cap .* inv_logit((k + A * delta) .* (t - (m + A * gamma)));
  }

  // Linear trend function

  vector linear_trend(
    real k,
    real m,
    vector delta,
    vector t,
    matrix A,
    vector t_change
  ) {
    return (k + A * delta) .* t + (m + A * (-t_change .* delta));
  }

  // Flat trend function

  vector flat_trend(
    real m,
    int T
  ) {
    return rep_vector(m, T);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// DATA COLLECTED BY THE MODEL
data {
  int T;                // Number of time periods of the time series
  int<lower=1> K;       // Number of seasonality regressors (dimension of vector X(t) K = 2N)
  vector[T] t;          // Vector of Times of the time series
  vector[T] cap;        // Capacities for logistic trend evolving in time
  vector[T] y;          // Time series values
  int S;                // Number of changepoints
  vector[S] t_change;   // Vector of Times of trend changepoints
  matrix[T,K] X;        // Seasonality Regressor vector for each time t
  vector[K] sigmas;     // Hyper-parameter of Scale on seasonality prior
  real<lower=0> tau;    // Hyper-parameter of Scale on changepoints prior
  int trend_indicator;  // 0 for linear, 1 for logistic, 2 for flat growth trend
  vector[K] s_a;        // Indicator of additive features
  vector[K] s_m;        // Indicator of multiplicative features
  // int<lower=1> T_pred;
}

transformed data {
  matrix[T, S] A;
  A = get_changepoint_matrix(t, t_change, T, S);    // to obtain the changepoint matrix a(t) for all t
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// PARAMETERS TO BE ESTIMATED BY THE MODEL
parameters {
  real k;                   // Base trend growth rate
  real m;                   // Trend offset
  vector[S] delta;          // Trend rate adjustments
  real<lower=0> sigma_obs;  // Observation noise
  vector[K] beta;           // Seasonality Regressor coefficients
}

transformed parameters {
  vector[T] trend;
  if (trend_indicator == 0) {
    trend = linear_trend(k, m, delta, t, A, t_change);
  } else if (trend_indicator == 1) {
    trend = logistic_trend(k, m, delta, t, cap, A, t_change, S);
  } else if (trend_indicator == 2) {
    trend = flat_trend(m, T);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// PRIORS AND LIKELIHOOD
model {
  //priors
  k ~ normal(0, 5);                      // prior for Base trend growth rate
  m ~ normal(0, 5);                      // prior Trend offset
  delta ~ double_exponential(0, tau);    // prior Trend rate adjustments (Laplace is double exponential)
  sigma_obs ~ normal(0, 0.5);            // prior for Observation noise
  beta ~ normal(0, sigmas);              // prior for Seasonality Regressor coefficients

  // Likelihood
  y ~ normal(
  trend
  .* (1 + X * (beta .* s_m))
  + X * (beta .* s_a),
  sigma_obs
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// GENERATED QUANTITIES
generated quantities {                       // vector containing the fitted values in the time range of
        vector[T] y_pred;           // the time series + space for following next 50 predictions             
        y_pred[1:T] = y;                     // collect the observed data in the time range of the series
}


