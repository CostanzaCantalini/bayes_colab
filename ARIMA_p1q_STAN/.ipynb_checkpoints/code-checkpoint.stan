

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
  vector<lower=-1,upper=1>[p] phi;
  vector<lower=-1,upper=1>[q] theta;
  real<lower=0> sigma;
  vector[p+1] y_start;
  vector[T] y_missing;
}
    
model {
  vector[T] nu;
  vector[T] err;
                
  for (t in 1:T) {

    nu[t] = 0.0;

    for (j in 1:p) {
      nu[t] += phi[j]*f(t-j,p,q,y,is_missing,y_missing,y_start);
    }

    for (j in 1:q) {
      nu[t] += theta[j]*g(t-j,err);
    }
    
    err[t] = f(t,p,q,y,is_missing,y_missing,y_start) - nu[t];
  }

  phi ~ normal(0, 2) T[-1,1];
  theta ~ normal(0, 2) T[-1,1];
  sigma ~ cauchy(0, 5) T[0,];
  err ~ normal(0, sigma);
        
  y_start ~ normal(1,1);
  y_missing ~ normal(1,1);
}


    
    
