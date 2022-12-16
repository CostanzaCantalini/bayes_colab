

functions {
  real f(int i, int p, int q, array[] real y, array[] int is_missing, vector y_missing, vector y_start) {

    if(i < 1) {
      return y_start[p+1+i] - y_start[p+i];
    }
    
    if(i == 1) {
      if(is_missing[1] == 1) {
	return y_missing[1] - y_start[p+1];
      } else {
	return y[1] - y_start[p+1];
      }                
    }
        
    if(is_missing[i] == 1) {
      if(is_missing[i-1] == 1) {
	return y_missing[i] - y_missing[i-1];
      } else {
	return y_missing[i] - y[i-1];
      }
    } else {
      if(is_missing[i-1] == 1) {
	return y[i] - y_missing[i-1];
      } else {
	return y[i] - y[i-1];
      }                
    }
  }

  real g(int i, vector err) {
    if (i < 1) {
      return 0.0;
    }
    return err[i];
  }
    
}

data {
  int<lower=1> T;
  int<lower=1> p;
  int<lower=1> q;
  array[T] real y;
  array[T] int<lower=0,upper=1> is_missing;
}
    
parameters {
  vector[q] gamma_th;
  vector[p] gamma_phi;
  real<lower=0> sigma;
  vector[p+1] y_start;
  vector[T] y_missing;
}

transformed parameters {
//  vector<lower=-2,upper=2>[q] theta;
  vector<lower=-1,upper=1>[q] theta;
  vector<lower=-1,upper=1>[p] phi;
  for (j in 1:q) {
//    theta[j] = 2*(exp(gamma_th[j]) - 1)/(exp(gamma_th[j]) + 1);
    theta[j] = 1*(exp(gamma_th[j]) - 1)/(exp(gamma_th[j]) + 1);
  }
  for (j in 1:p) {
    phi[j] = (exp(gamma_phi[j]) - 1)/(exp(gamma_phi[j]) + 1);
  }

  //  array[T+p+1] real right_y = y;
  vector[T+p+1] right_y;
  for (i in 1:(p+1)) {
    right_y[i] = y_start[i];
  }
  for (i in 1:T) {
    if (is_missing[i] == 1) {
      right_y[i+p+1] = y_missing[i];
    } else {
      right_y[i+p+1] = y[i];
    }
  }
  //  array[T+p] real D = right_y[2:(T+p+1)] - right_y[1:(T+p)];
  vector[T+p] D = right_y[2:(T+p+1)] - right_y[1:(T+p)];
}
    
model {
  vector[T] nu;
  vector[T] err;
                
  for (t in 1:T) {

    nu[t] = 0.0;

    for (j in 1:p) {
      nu[t] += phi[j]*D[t-j+p];
    }

    for (j in 1:q) {
      nu[t] += theta[j]*g(t-j,err);
    }
    
    err[t] = D[t+p] - nu[t];
  }

  //phi ~ normal(0, 2) T[-1,1];
  gamma_phi ~ normal(0,1);
  //theta ~ normal(0, 2) T[-1,1];
  gamma_th ~ normal(0,1);
  //gamma_th ~ normal(0,0.67);
  sigma ~ cauchy(0, 5) T[0,];
  err ~ normal(0, sigma);
        
  y_start ~ normal(1,1);
  y_missing ~ normal(1,1);
}


    
generated quantities {
    array[T] real y_post_pred;
    array[T] real err_post_pred = normal_rng(rep_vector(0,T), sigma);
    for (t in 1:T) {
        
        real mean_val = 0.0;
        
        for (j in 1:p) {
            if (t-j < 1) {
                mean_val += phi[j]*(y_start[p+t-j+1] - y_start[p+t-j]);
            } else if ((t-j)==1) {
                mean_val += phi[j]*(y_post_pred[t-j] - y_start[p+t-j]);
            } else {
                mean_val += phi[j]*(y_post_pred[t-j] - y_post_pred[t-j-1]);
            }
        }
    
        for (j in 1:q) {
            if (t-j > 0) {
                mean_val += theta[j]*err_post_pred[t-j];
            }
        }

        if (t == 1) {
            y_post_pred[t] = y_start[p+1] + mean_val + err_post_pred[t];            
        } else {
            y_post_pred[t] = y_post_pred[t-1] + mean_val + err_post_pred[t];
        }
        
    }
}

