
data {

  int<lower=1> T;	// la lunghezza dell'intervallo temporale

  int<lower=1> p;	// il grado della porzione autoregressiva
  int<lower=1> q;	// il grado della porzione a media mobile

  vector[T] y;		// i dati osservati (al posto dei NaN si passa 1.0 per compatibilità,
  			//ma il valore non viene realmente utilizzato)

  vector<lower=0,upper=1>[T] is_missing; // contiene un 1 se il dato corrispondente è mancante, altrimenti 0

}
    
parameters {

  vector[q] gamma_th;	// "logit"(theta)
  vector[p] gamma_phi;	// "logit"(phi)

  // varianza di err
  real<lower=0> sigma;

  // dati mancanti
  vector[p+1] y_start;
  vector[T] y_missing;
}

transformed parameters {

  // theta e phi sono ottenuti mediante:
  //            exp(gamma_th) - 1  
  //   theta = --------------------
  //            exp(gamma_th) + 1
  // analogamente per phi, nelle righe a seguire
  vector<lower=-1,upper=1>[q] theta;
  vector<lower=-1,upper=1>[p] phi;
  for (j in 1:q) {
    theta[j] = (exp(gamma_th[j]) - 1)/(exp(gamma_th[j]) + 1);
  }
  for (j in 1:p) {
    phi[j] = (exp(gamma_phi[j]) - 1)/(exp(gamma_phi[j]) + 1);
  }

  
  // produce un vettore che contiene il dato
  // in corrispondenza dei dati noti, e contiene una variabile y_missing oppure y_start
  // in corrispondenza dei dati mancanti
  // poi lo differenzia per produrre il vettore D (parte integrativa dell'ARIMA)
  vector[T+p+1] right_y;
  right_y[1:(p+1)] = y_start;
  right_y[(p+2):(T+p+1)] = y_missing.*is_missing + y.*(1-is_missing);
  vector[T+p] D = right_y[2:(T+p+1)] - right_y[1:(T+p)];  
}
    
model {

  vector[T] nu = rep_vector(0,T);
  vector[T] err;


  // AR VETTORIALIZZATO
  for (j in 1:p) {
    nu += phi[j]*D[(1-j+p):(T-j+p)];
  }
  
  // MA
  err = D[(p+1):(p+T)] - nu;
  for (t in 1:T) {
    for (j in 1:q) {
      if ((t-j) > 0) {
	err[t] -= theta[j]*err[t-j];
      }
    }
  }

  
  // priors
  gamma_phi ~ normal(0,1);
  gamma_th ~ normal(0,1);

  sigma ~ cauchy(0, 5) T[0,];
        
  y_start ~ normal(1,1);
  y_missing ~ normal(1,1);


  // likelihood
  err ~ normal(0, sigma);
  
}



generated quantities {
  
  // posterior predictive
  // TO-DO DA VETTORIALIZZARE (in realtà forse no perché sembra che non rallenta)
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

