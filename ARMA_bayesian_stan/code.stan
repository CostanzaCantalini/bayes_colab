

    functions {
        real f(int i, array[] real y, array[] int is_missing, vector y_missing, vector y_start) {
        
            if(i == 1) {
                if(is_missing[1] == 1) {
                    return y_missing[1] - y_start[3];
                } else {
                    return y[1] - y_start[3];
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
    }

    data {
        int<lower=1> T;
        array[T] real y;
        array[T] int<lower=0,upper=1> is_missing;
    }
    
    parameters {
        real<lower=-1,upper=1> phi;                  // autoregression coeff
        real<lower=-1,upper=1> theta;                // moving avg coeff
        real<lower=-1,upper=1> phi_2;                // autoregression coeff
        real<lower=0> sigma;                         // noise scale
        vector[3] y_start;
        vector[T] y_missing;
    }
    
    model {
        vector[T] nu;              // prediction for time t
        vector[T] err;             // error for time t
        
        
        if(is_missing[1] == 1){                    
            nu[1] = phi * (y_start[3] - y_start[2]) + phi_2 * (y_start[2] - y_start[1]);
            err[1] = (y_missing[1] - y_start[3]) - nu[1];

            if(is_missing[2] == 1){
                nu[2] = phi * (y_missing[1] - y_start[3]) + phi_2 * (y_start[3] - y_start[2]) + theta * err[1];
                err[2] = (y_missing[2] - y_missing[1]) - nu[2];
            } else {
                nu[2] = phi * (y_missing[1] - y_start[3]) + phi_2 * (y_start[3] - y_start[2]) + theta * err[1];
                err[2] = (y[2] - y_missing[1]) - nu[2];
            }

        } else {        
            nu[1] = phi * (y_start[3] - y_start[2]) + phi_2 * (y_start[2] - y_start[1]);
            err[1] = (y[1] - y_start[3]) - nu[1];        

            if(is_missing[2] == 1){
                nu[2] = phi * (y[1] - y_start[3]) + phi_2 * (y_start[3] - y_start[2]) + theta * err[1];
                err[2] = (y_missing[2] - y[1]) - nu[2];
            } else {
                nu[2] = phi * (y[1] - y_start[3]) + phi_2 * (y_start[3] - y_start[2]) + theta * err[1];
                err[2] = (y[2] - y[1]) - nu[2];
            }

        }

        for (t in 3:T) {
            nu[t] = phi * f(t-1,y,is_missing,y_missing,y_start) + phi_2 * f(t-2,y,is_missing,y_missing,y_start) + theta * err[t - 1];
            err[t] = f(t,y,is_missing,y_missing,y_start) - nu[t];
        }

        phi ~ normal(0, 2) T[-1,1];
        theta ~ normal(0, 2) T[-1,1];
        phi_2 ~ normal(0, 2) T[-1,1];
        sigma ~ cauchy(0, 5) T[0,];
        err ~ normal(0, sigma);
        
        y_start ~ normal(1,1);
        y_missing ~ normal(1,1);
    }

    generated quantities {
        array[T] int<lower=0,upper=1> missing_data = is_missing;
    }

    
    
