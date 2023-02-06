
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
  int T;                         // Number of time periods of the time series
  int<lower=1> K;                // Number of seasonality regressors (dimension of vector X(t) K = 2N)
  vector[T] t;                   // Vector of Times of the time series
  vector[T] cap;                 // Capacities for logistic trend evolving in time
  vector[T] y;                   // Time series values
  int S;                         // Number of changepoints
  vector[S] t_change;            // Vector of Times of trend changepoints
  matrix[T,K] X_1;               // Seasonality factor 1 Regressor vector for each time t
  vector[K] sigmas_1;            // Hyper-parameter of Scale on seasonality prior
  vector[K] sigmas_2;            // Hyper-parameter of Scale on seasonality prior
  real<lower=0> tau;             // Hyper-parameter of Scale on changepoints prior
  int trend_indicator;           // 0 for linear, 1 for logistic, 2 for flat growth trend
  vector[K] s_a;                 // Indicator of additive features for seasonality
  vector[K] s_m;                 // Indicator of multiplicative features for seasonality
  matrix[T,K] X_2;               // Seasonality factor 2 Regressor vector for each time t
  int<lower=1> T_pred;           // Number of time periods of the time series for prediction
  vector[T_pred] t_pred;         // Vector of Times of the time series for prediction
  vector[T_pred] cap_pred;       // Capacities for logistic trend evolving in time during prediction interval
  int S_pred;                    // Number of changepoints in prediction interval
  vector[S_pred] t_change_pred;  // Vector of Times of trend changepoints in prediction interval
  matrix[T_pred,K] X_pred_1;     // Seasonality factor 1 Regressor vector for each time t in prediction interval
  matrix[T_pred,K] X_pred_2;     // Seasonality factor 2 Regressor vector for each time t in prediction interval
}

transformed data {
  matrix[T, S] A;
  matrix[T_pred, S] A_pred;
  A = get_changepoint_matrix(t, t_change, T, S);    // to obtain the changepoint matrix a(t) for all t
  A_pred = get_changepoint_matrix(t_pred, t_change_pred, T_pred, S_pred);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// PARAMETERS TO BE ESTIMATED BY THE MODEL
parameters {
  real k;                    // Base trend growth rate
  real m;                    // Trend offset
  vector[S] delta;           // Trend rate adjustments
  real<lower=0> sigma_obs;   // Observation noise
  vector[K] beta_1;          // Seasonality Regressor coefficients
  vector[K] beta_2;          // Second Seasonality Regressor coefficients
}

transformed parameters {     // select the growht rate type
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
  beta_1 ~ normal(0, sigmas_1);          // prior for Seasonality factor 1 Regressor coefficients
  beta_2 ~ normal(0, sigmas_2);          // prior for Seasonality factor 2 Regressor coefficients

  // Likelihood
  y ~ normal(
  trend .* (1 + X_1 * (beta_1 .* s_m)) + X_1 * (beta_1 .* s_a) +
  trend .* (1 + X_2 * (beta_2 .* s_m)) + X_2 * (beta_2 .* s_a),
  sigma_obs
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// GENERATED QUANTITIES
generated quantities {                       // vector containing the fitted values in the time range of
        vector[T+T_pred] y_pred;             // the time series + space for following next 50 predictions             
        y_pred[1:T] = y;                     // collect the observed data in the time range of the series
        vector[T_pred] trend_pred;           // collect the trend growth rates in the prediction interval
        if (trend_indicator == 0) {          // select the type of growth rate
            trend_pred = linear_trend(k, m, delta, t_pred, A_pred, t_change_pred);
        } else if (trend_indicator == 1) {
            trend_pred = logistic_trend(k, m, delta, t_pred, cap_pred, A_pred, t_change_pred, S_pred);
        } else if (trend_indicator == 2) {
            trend_pred = flat_trend(m, T_pred);
        }
        y_pred[(T+1):(T+T_pred)] = trend_pred.* (1 + X_pred_1 * (beta_1 .* s_m)) + X_pred_1 * (beta_1 .* s_a)
                                 + trend_pred.* (1 + X_pred_2 * (beta_2 .* s_m)) + X_pred_2 * (beta_2 .* s_a);
}


