library(readxl)

data<-read_excel("C:/Users/enogj/Documents/GitHub/bayes_colab/data/dati_polveri/Polveri Emilia.xlsx", 
                 sheet = 1)

# data<- data[,-10]
attach(Polveri_Emilia)
anno<-factor(Anno)

data_2018<- Polveri_Emilia[which(anno=='2018'),]
# data_2018<- data_2018[,-2]
detach(data)

table(data_2018$VALORE)
data_2018$VALORE[data_2018$VALORE == 0] <- 1


#log transform of values

# data_2018$VALORE<-log(data_2018$VALORE)
# hist(data_2018$VALORE)


# Create date
data_2018$date <- as.Date(with(data_2018, paste(Anno, data_2018$Mese, data_2018$Giorno,sep="-")), "%Y-%m-%d")
data_2018$date

# Choose only one station for simplicity, BOGOLESE
stat = data_2018[data_2018$COD_STAZ == 2000230,]

#missing dates

date_range <- seq(min(stat$date), max(stat$date), by = 1)
date_range[!date_range %in% stat$date]

#add missing dates
library(padr)

padded_df <- pad(stat, interval = "day", end_val = stat$date[350])
padded_df <- fill_by_value(padded_df,VALORE, value =NA )
padded_df

padded_df$VALORE[which(is.na(padded_df$VALORE))] = 1

df = padded_df
df.log = df 
df.log$VALORE = log(df.log$VALORE)

library(bayesforecast)
library(xts)

# test = ts(data_2018$VALORE)
# autoplot(test)
# 
# val.ts <- xts(df$VALORE, order.by=df$date)
# autoplot(val.ts)
# 
val.tsl <- xts(df.log$VALORE, order.by=df.log$date)
autoplot(val.tsl)

# Run bayesian ARIMA estimation


## FRIST ATTEMPT: Default parameters

# # Estimated model for raw values
# res = auto.sarima(val.ts) 
# 
# autoplot(res)
# 
# fitted(res)
# 
# # Estimated model for log values
# resl = auto.sarima(val.tsl) 
# 
# autoplot(resl)
# 
# fitted(resl)

## SECOND ATTEMPT

# res1 = auto.sarima(val.ts,seasonal = TRUE,chains=1,stepwise = FALSE,trace=TRUE,approximation= FALSE,nmodels=94,stationary = FALSE,
#                    max.p = 1,
#                    max.q = 1,
#                    max.d = 5, # Maximum number of non-seasonal differences
#                    max.D = 2)# Maximum number of seasonal differences
# 
# library(forecast)
# fit = auto.arima(val.tsl,seasonal = FALSE,stepwise = FALSE,trace=TRUE,approximation= FALSE)
# 
# autoplot(res1)
# mcmc_plot(res1)

resl1 = auto.sarima(val.tsl,seasonal = TRUE,chains=1,stepwise = FALSE,trace=TRUE,approximation= FALSE,nmodels=94,
                    max.p = 3,
                    max.q = 3)# Maximum number of seasonal differences

autoplot(resl1)
mcmc_plot(resl1)
# This line creates a list with the data for running stan() rstan package
#resl2 = Sarima(val.tsl,order = c(0,1,3),seasonal = c(0,0,0),xreg = NULL,period = 0,series.name = NULL) 

# Checking all models through auto.sarima
vec <- matrix(0,49,4)
stazione = as.factor(data_2018$COD_STAZ)
for (ii in 1:length(levels(stazione))){
  staz <- xts(log(data_2018[which(stazione==levels(stazione)[ii]),]$VALORE), order.by=data_2018[which(stazione==levels(stazione)[ii]),]$date)
  fit <- auto.sarima(staz,seasonal=FALSE,stepwise=FALSE,approximation=FALSE,chains = 1,iter=10)
  model = paste(as.character(fit$model$p),as.character(fit$model$d),as.character(fit$model$q))
  vec[ii,]<-c(model,unique(data_2018[which(stazione==levels(stazione)[ii]),]$TipoStazione),unique(data_2018[which(stazione==levels(stazione)[ii]),]$TipoArea),levels(stazione)[ii])
  summary(fit)
  print(ii)
  }

library(dplyr)
vec<-as.data.frame(vec)
distinct(vec[,-4])
# vec$V4 = as.factor(vec$V4)
# vec$V5 = as.factor(vec$V5)
# vec$V1 = as.character(vec$V1)

# Order by TipoArea
model_area = vec %>%
  group_by(V1,V3) %>%
  summarise(total_count=n(),.groups = 'drop') %>%
  arrange(V3) %>%
  as.data.frame()
model_area

# Order by TipoStazione
model_tipo = vec %>%
  group_by(V1,V2) %>%
  summarise(total_count=n(),.groups = 'drop') %>%
  arrange(V2) %>%
  as.data.frame()
model_tipo

# Order by TipoArea and TipoStazione
model_tipo_area = vec %>%
  group_by(V1,V2,V3) %>%
  summarise(total_count=n(),.groups = 'drop') %>%
  arrange(V2,V3) %>%
  as.data.frame()
model_tipo_area

