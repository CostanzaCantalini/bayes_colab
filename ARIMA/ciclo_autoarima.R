vec <- matrix(0,49,3)
Stazione<-as.factor(Stazione)
stazione<-levels(Stazione)
library(forecast)

data_2018$VALORE<-log(data_2018$VALORE)

for (ii in 1:length(levels(stazione)))
{staz <- data_2018[which(Stazione==levels(stazione)[ii]),]$VALORE
fit <- auto.arima(staz,stepwise=FALSE,approximation=FALSE)
vec[ii,]<-c(length(fit$model$phi),length(fit$model$Delta),length(fit$model$theta))
summary(fit)}

####
# controllo non-cancellazione zeri e poli
l<-list("prova")
l1<-list("prova1")
for (ii in 1:length(stazione))
{staz <- na.omit(data_2018[which(Stazione==stazione[ii]),]$VALORE)
fit <- auto.arima(staz,stepwise=FALSE,approximation=FALSE,d=1)
#arma.roots(c(fit$model$phi,fit$model$theta))
l1<-c(fit$model$phi,fit$model$theta, length(fit$model$phi), length(fit$model$theta))
l[ii]<-list(l1)}

library(dplyr)
vec<-as.data.frame(vec)
distinct(vec)
group_by(vec)


tab<-rep(0,10)
vec2<-rep(0,49)
for (jj in 1:10)
{ for (ii in 1:49)
{tab[jj]<-tab[jj]+all(vec[ii,]==distinct(vec)[jj,])
  if (all(vec[ii,]==distinct(vec)[jj,]))
vec2[ii]<-jj}}
tab
vec2
sum(tab)

for (ii in 1:49)
{which(all(vec[ii,]==distinct(vec)[2,]))}
vec2

col=vec2

plot1 =  ggplot(urban_data, aes(x=date, y=VALORE), col=col) +
  geom_line() + 
  xlab("urban")
plot2 = ggplot(suburban_data, aes(x=date, y=VALORE), col=col) +
  geom_line() + 
  xlab("suburban")
plot3 = ggplot(rural_data, aes(x=date, y=VALORE), col=col) +
  geom_line() + 
  xlab("rural")
grid.arrange(plot1, plot2, plot3, ncol=3, nrow = 1)
