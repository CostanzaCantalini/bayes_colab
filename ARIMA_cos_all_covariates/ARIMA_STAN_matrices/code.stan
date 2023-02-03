
data {

  int<lower=1> T;	// la lunghezza dell'intervallo temporale
  int<lower=1> S;	// il numero di stazioni
  int<lower=1> reg; //numero di regressori utilizzati

  int<lower=1> p;	// il grado della porzione autoregressiva
  int<lower=1> q;	// il grado della porzione a media mobile

  matrix[T,S] y;	// i dati osservati (al posto dei NaN si passa 1.0 per compatibilità,
  			//ma il valore non viene realmente utilizzato)
  matrix[S,reg] X;

  matrix<lower=0,upper=1>[T,S] is_missing; // contiene un 1 se il dato corrispondente è mancante, altrimenti 0

  int<lower=0> missing_size;	// contiene la somma di is_missing (serve farlo esternamente per caratteristiche specifiche di stan)

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

  
}

parameters {
  
  //real alpha; //Intercept
  //real beta; //Slope
  vector [reg] betas;
  
  vector[q] gamma_th;		// "logit"(theta)
  array[p] row_vector[S] gamma_phi;	// "logit"(phi)

  // varianza di err
  real<lower=0> sigma; // forse deve diventare una funzione della stazione (?)

  // dati mancanti
  matrix[p+1,S] y_start;
  //  matrix[T,S] y_missing;
  vector[missing_size] w;

//  real<lower=0> c;
  vector[S] c;
//  real delta;

  vector[p+1] hyper_y_start_m;
  vector<lower=0>[p+1] hyper_y_start_s;

  vector[p] hyper_gamma_phi_m;
  vector<lower=0>[p] hyper_gamma_phi_s;
}

transformed parameters {

  //matrix[T,S] cos_of_day = c*rep_matrix(cos( 2*pi()*(cumulative_sum(rep_vector(1,T)) + delta)/365 ) , S);
  /*
  matrix[T,S] cos_of_day;
  for (s in 1:S) {
    cos_of_day[:,s] = c[s]*cos( 2*pi()*(cumulative_sum(rep_vector(1,T)) + delta)/365 );
  }
  */
  matrix[T,S] cos_of_day;
  for (s in 1:S) {
    cos_of_day[:,s] = c[s]*cos( 2*pi()*(cumulative_sum(rep_vector(1,T))  )/365 );
  }
  
  matrix[T,S] regres;
  for (s in 1:S) {
    real coeff=0;
    for (r in 1:reg){
      
      coeff+=X[s,r]*betas[r];
    }
    
    for (t in 1:T){
      regres[t,s] = coeff;
    }
    
  }
  
  // theta e phi sono ottenuti mediante:
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
  // in corrispondenza dei dati noti, e contiene una variabile y_missing oppure y_start
  // in corrispondenza dei dati mancanti
  // poi lo differenzia per produrre il vettore D (parte integrativa dell'ARIMA)
  matrix[T+p+1,S] right_y;
  right_y[1:(p+1), 1:S] = y_start ;//bisognerebbe togliere cos
  right_y[(p+2):(T+p+1), 1:S] = y - cos_of_day- regres; 
  for (k in 1:missing_size) {
    right_y[p+1+u_mine[k],v[k]] = w[k] ;//bisognerebbe togliere cos
  }
  matrix[T+p, S] D = right_y[2:(T+p+1), 1:S] - right_y[1:(T+p), 1:S];  
}
    
model {

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

  
  // priors
  //gamma_phi ~ multi_normal(rep_vector(0,S),diag_matrix(rep_vector(1,S)));
  for (j in 1:p) {
    gamma_phi[j] ~ normal(hyper_gamma_phi_m[j], hyper_gamma_phi_s[j]);
  }
  gamma_th ~ normal(0,1);

  sigma ~ cauchy(0, 5) T[0,];
        
  for (j in 1:(p+1)) {
    y_start[j,1:S] ~ normal(hyper_y_start_m[j],hyper_y_start_s[j]);
  }
  w ~ normal(1,1);

  hyper_y_start_m ~ normal(1,1);
  hyper_y_start_s ~ inv_gamma(3,2);

  hyper_gamma_phi_m ~ normal(0,5);
  hyper_gamma_phi_s ~ inv_gamma(2.1,1.1);

  //c ~ normal(0,1) T[0,];
  c ~ normal(0,1);
  //delta ~ normal(0,10);
  betas ~ normal(0,1);
  
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
  y_post_pred_aux[1:(p+1), 1:S] = y_start;#bisognerebbe aggiungere cos

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

  
  matrix[T,S] y_post_pred = y_post_pred_aux[p+2:T+p+1,:] + cos_of_day[:,:]+regres[:,:]; //aggiungere parte regressiva
  
}
