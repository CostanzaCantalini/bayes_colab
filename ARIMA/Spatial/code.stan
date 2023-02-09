
data {

  int<lower=1> T;	// la lunghezza dell'intervallo temporale
  int<lower=1> S;	// il numero di stazioni
  int<lower=1> reg;	//numero di regressori utilizzati

  int<lower=1> p;	// il grado della porzione autoregressiva
  int<lower=1> q;	// il grado della porzione a media mobile

  matrix[T,S] y;	// i dati osservati (al posto dei NaN si passa 1.0 per compatibilità,
  			//ma il valore non viene realmente utilizzato)
  matrix[S,reg] X;

  matrix<lower=0,upper=1>[T,S] is_missing; // contiene un 1 se il dato corrispondente è mancante, altrimenti 0

  int<lower=0> missing_size;	// contiene il numero di dati mancanti

  array[S] vector[2] coord;
  
}

transformed data {
  
  array[T+1] int u;
  array[missing_size] int<lower=1,upper=S> v;

  u = csr_extract_u(is_missing);
  v = csr_extract_v(is_missing);

  array[missing_size] int<lower=1,upper=T> u_mine;

  int index = 1;
  for (t in 1:T) {
    if (u[t+1]-u[t] > 0) {
      for (j in 1:(u[t+1]-u[t])) {
	u_mine[index] = t;
	index += 1;
      }
    }
  }

  row_vector[S] media_stazione;
  for (s in 1:S) {
    media_stazione[s] = mean(y[1:T,s]);
  }

  real rho = 0.05;

  vector[p+1] y_start = rep_vector(mean(media_stazione),p+1);
  
}

parameters {
  
  // Regressione
  vector[reg] betas;

  // ARMA
  vector[q] gamma_th;			// "logit"(theta)
  array[p] row_vector[S] gamma_phi;	// "logit"(phi)
  vector[p] hyper_gamma_phi_m;
  vector<lower=0>[p] hyper_gamma_phi_s;

  // deviazione standard di err e iperparametri associati
  real<lower=0> sigma;
  real mh_sigma;
  real<lower=0> sh_sigma;

  // dati mancanti
  vector[missing_size] w;

  // Coefficiente moltiplicativo del coseno
  vector[S] c;

  // Residui spaziali e deviazione standard del residuo spaziale
  vector[S] w_s;
  real<lower=0> alpha ;

}

transformed parameters {

  matrix[T,S] cos_of_day;
  for (s in 1:S) {
    cos_of_day[:,s] = c[s]*cos( 2*pi()*(cumulative_sum(rep_vector(1,T)) )/365 );
  }
  
  matrix[T,S] regres = rep_matrix((X*betas)',T);

  cov_matrix[S] H = gp_exp_quad_cov(coord,alpha,rho);  
  matrix[T,S] spatial = rep_matrix(w_s',T);

  
  // THETA e phi sono ottenuti mediante:
  //            exp(gamma_th) - 1  
  //   theta = --------------------
  //            exp(gamma_th) + 1
  // analogamente per phi, nelle righe a seguire
  vector<lower=-1,upper=1>[q] theta;
  array[p] row_vector<lower=-1,upper=1>[S] phi;
  for (j in 1:q) {
    theta[j] = (exp(gamma_th[j]) - 1)/(exp(gamma_th[j]) + 1);
  }
  for (j in 1:p) {
    phi[j] = (exp(gamma_phi[j]) - 1)./(exp(gamma_phi[j]) + 1);
  }

  // produce un vettore che contiene il dato
  // in corrispondenza dei dati noti, e contiene una variabile w oppure y_start
  // in corrispondenza dei dati mancanti
  // poi lo differenzia per produrre il vettore D (parte integrativa dell'ARIMA)
  matrix[T+p+1,S] right_y;
  right_y[1:(p+1), 1:S] = rep_matrix(y_start,S);
  right_y[(p+2):(T+p+1), 1:S] = y; 
  for (k in 1:missing_size) {
    right_y[p+1+u_mine[k],v[k]] = w[k];
  }
  right_y[(p+2):(T+p+1), 1:S] = right_y[(p+2):(T+p+1), 1:S] - cos_of_day - regres - spatial; 
  matrix[T+p, S] D = right_y[2:(T+p+1), 1:S] - right_y[1:(T+p), 1:S];  


  // ARIMA calcolo dei residui
  matrix[T,S] nu = rep_matrix(0,T,S);
  matrix[T,S] err;

  // AR VETTORIALIZZATO
  for (j in 1:p) {
    nu += rep_matrix(phi[j],T).*D[(1-j+p):(T-j+p), 1:S];
  }
  
  // MA
  err = D[(p+1):(p+T), 1:S] - nu;
  for (t in 1:T) {
    for (j in 1:q) {
      if ((t-j) > 0) {
	err[t, 1:S] -= theta[j]*err[t-j, 1:S];
      }
    }
  }


}
    
model {

  
  // priors
  for (j in 1:p) {
    gamma_phi[j] ~ normal(hyper_gamma_phi_m[j], hyper_gamma_phi_s[j]);
  }
  gamma_th ~ normal(0,1);
  hyper_gamma_phi_m ~ normal(0,5);
  hyper_gamma_phi_s ~ inv_gamma(3,2);

  
  sigma ~ lognormal(mh_sigma, sh_sigma);
  mh_sigma ~ normal(0,1);
  sh_sigma ~ inv_gamma(3,2);

  
  w ~ normal(mean(media_stazione),1);



  c ~ normal(0,1);



  betas ~ normal(0,1);

  alpha ~ inv_gamma(6, 2);
  w_s ~ multi_normal(rep_vector(0,S), H);

  // likelihood
  for (s in 1:S) {
    err[1:T,s] ~ normal(0, sigma);
  }

}


generated quantities {

  matrix[T+p+1,S] y_post_pred_aux;
  matrix[T,S] err_post_pred;
  for (s in 1:S) {
    err_post_pred[1:T,s] = to_vector(normal_rng(rep_vector(0,T), sigma));
  }
  y_post_pred_aux[1:(p+1), 1:S] = rep_matrix(mean(media_stazione),(1+p),S);

  for (t in (p+2):(T+p+1)) {
    
    row_vector[S] mean_val = rep_row_vector(0,S);
        
    for (j in 1:p) {
      mean_val += phi[j].*(y_post_pred_aux[t-j, :] - y_post_pred_aux[t-j-1, :]);
    }
    
    for (j in 1:q) {
      if ((t-p-1)-j > 0) {
	mean_val += theta[j]*err_post_pred[(t-p-1)-j, :];
      }
    }

    y_post_pred_aux[t,:] = y_post_pred_aux[t-1,:] + mean_val + err_post_pred[t-p-1,:];
        
  }

  
  matrix[T,S] y_post_pred = y_post_pred_aux[p+2:T+p+1,:] + cos_of_day[:,:] + regres[:,:] + spatial[:,:];

  vector[S] annual_mean;
  vector[S] annual_max;
  vector[S] annual_median;
  vector[S] annual_days_over_threshold;
  array[S] int is_over_daily_limit;
  array[S] int is_over_annual_limit;
  for (s in 1:S) {
    annual_mean[s] = mean(y_post_pred[:,s]);
    annual_max[s] = max(y_post_pred[:,s]);
    annual_median[s] = quantile(y_post_pred[:,s],0.5);
    annual_days_over_threshold[s] = 0;
    for (t in 1:T) {
      annual_days_over_threshold[s] += (y_post_pred[t,s] > log10(50));
    }
    is_over_daily_limit[s] = (annual_days_over_threshold[s] > 35);
    is_over_annual_limit[s] = (annual_mean[s] > log10(40));
  }
    


  vector[T*S-missing_size] log_lik;
  int index_gen = 1;
  for (t in 1:T) {
    for (s in 1:S) {
      if (is_missing[t,s] == 0) {
	log_lik[index_gen] = normal_lpdf(err[t,s] | 0, sigma);
	index_gen += 1;
      }
    }
  }

  
  array[missing_size] int<lower=1,upper=T> missing_index_time = u_mine;
  array[missing_size] int<lower=1,upper=S> missing_index_station = v;
  
}
