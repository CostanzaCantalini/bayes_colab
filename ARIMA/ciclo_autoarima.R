install.packages("RColorBrewer")

vec <- matrix(0,49,3)
for (ii in 1:length(levels(stazione)))
{staz <- data_2018[which(Stazione==levels(stazione)[ii]),]$VALORE
fit <- auto.arima(staz,stepwise=FALSE,approximation=FALSE)
vec[ii,]<-c(length(fit$model$phi),length(fit$model$Delta),length(fit$model$theta))
summary(fit)}

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

library(RColorBrewer)
col=vec2
library(gg)

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
