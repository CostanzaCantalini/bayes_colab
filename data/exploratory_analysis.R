library(readxl)

data<-read_excel("Polveri Emilia.xlsx", 
                      sheet = 1)

data<- data[,-10]
attach(data)
anno<-factor(Anno)

data_2018<- data[which(anno=='2018'),]
data_2018<- data_2018[,-2]
detach(data)
attach(data_2018)


#replace 0 values with 1
table(data_2018$VALORE)
data_2018$VALORE[data_2018$VALORE == 0] <- 1

#log transform of values

data_2018$VALORE<-log(data_2018$VALORE)
hist(data_2018$VALORE)

stazione<-factor(Stazione)

data_2018$date <- as.Date(with(data_2018, paste(Anno, data_2018$Mese, data_2018$Giorno,sep="-")), "%Y-%m-%d")
data_2018$date
x11() 
plot(data_2018$date,data_2018$VALORE,xlab='Days',ylab='Values') #useless


library(ggplot2)
library(dplyr)


# Most basic time series plot
p <- ggplot(data.frame(
  day = data_2018$date,
  value = data_2018$VALORE
), aes(x=day, y=value)) +
  geom_line() + 
  xlab("")
p


boxplot(data_2018$VALORE~stazione,data_2018,ylab = 'values') #boxplot for stations
boxplot(data_2018$VALORE) #boxplot of all values
boxplot(data_2018$VALORE~type) #boxplot on type
boxplot(data_2018$VALORE~area) #boxplot on area


type<-factor(data_2018$TipoStazione)
area<- factor(data_2018$TipoArea)

library(directlabels)
library(ggplot2)

#weekend variable


is_weekend <- function(n) {
  require(lubridate)
  
  (ifelse(wday(as.Date(n)) == 1, T, F) | ifelse(wday(as.Date(n)) == 7, T, F))
  
}   
data_2018$weekend<-is_weekend(data_2018$date)

#datasets based on area

urban_data<- data_2018[which(area=='Urbano'),]
suburban_data<-  data_2018[which(area=='Suburbano'),]
rural_data<-  data_2018[which(area=='Rurale'),]
x11()
par(mfrow=c(1,3))
 ggplot(urban_data, aes(x=date, y=VALORE)) +
  geom_line() + 
  xlab("")
 ggplot(suburban_data, aes(x=date, y=VALORE)) +
   geom_line() + 
   xlab("")
 x11()
 ggplot(rural_data, aes(x=date, y=VALORE)) +
   geom_line() + 
   xlab("")



#plots of values divided by area and type



ggplot(data_2018, aes(date, VALORE, group = area,color=area)) + 
  geom_line() +
  #geom_point() +
  geom_dl(aes(label = area), 
          method = list(dl.combine("first.points", "last.points"), cex = 0.8)) 
scale_color_gradient2()

ggplot(data_2018, aes(date, VALORE, group = type,color=type)) + 
  geom_line() +
  #geom_point() +
  geom_dl(aes(label = type), 
          method = list(dl.combine("first.points", "last.points"), cex = 0.8)) 
scale_color_gradient2()


# mu<-tapply(data_2018$VALORE, stazione, mean)
# S<- tapply(data_2018$VALORE,stazione,sd)
# 
# 
# days<- seq( as.Date("2018-01-01"), as.Date("2018-12-31"), by="+1 day")
# 
# df<- data.frame(data_2018$VALORE)
# 
# iterations <- 365
# variables<- 49
# 
# output <- matrix(ncol=variables, nrow=iterations)
# 
#  for(i in 1:variables){
#    for(j in 1:iterations){
#      
#      output[i,j] <- data_2018[which(stazione==levels(stazione)[j]),]$VALORE[i]
#    }
#  }

#some stations
staz_1<-data_2018[which(stazione==levels(stazione)[1]),]





ggplot(staz_1, aes(x=date, y=VALORE)) +
  geom_line() + 
  xlab("")

staz_2<-data_2018[which(stazione==levels(stazione)[2]),]




ggplot(staz_2, aes(x=date, y=VALORE)) +
  geom_line() + 
  xlab("")

library(forecast)

#auto arima on station 1

fit <- auto.arima(staz_1$VALORE,stepwise=FALSE,approximation=FALSE)
summary(fit)

# Next 5 forecasted values
forecast(fit, 5)

# plotting the graph with next
# 5 weekly forecasted values
plot(forecast(fit, 5), xlab ="Days",
     ylab ="PM10 values",
     main ="Badia", col.main ="darkgreen")


#library(xts)



# Plot station 1 vs station 2
ggplot() + 
  geom_line(data = staz_1, aes(x = date, y = VALORE), color = "red") +
  geom_line(data = staz_2, aes(x = date, y = VALORE), color = "blue") +
  xlab('days') +
  ylab('PM10')


staz_3<-data_2018[which(stazione==levels(stazione)[3]),]





ggplot(staz_3, aes(x=date, y=VALORE)) +
  geom_line() + 
  xlab("")

#station 1 vs station 3

ggplot() + 
  geom_line(data = staz_1, aes(x = date, y = VALORE), color = "red") +
  geom_line(data = staz_3, aes(x = date, y = VALORE), color = "blue") +
  xlab('days') +
  ylab('PM10')

library(mice)

#md.pattern(padded_df,plot=TRUE)

#stationariety

library(urca)

summary(ur.kpss(staz_1$VALORE)) #stationariety test
ndiffs(staz_1$VALORE) #seems stationary, number of differentiations needed

#nsdiffs(staz_3$VALORE)


summary(ur.kpss(staz_2$VALORE))
ndiffs(staz_2$VALORE) #not stationary 1 diff

summary(ur.kpss(staz_3$VALORE))
ndiffs(staz_3$VALORE) #not stationary  1 diff

#pacf and acf plots

ggtsdisplay(staz_1$VALORE)

#try some arima fits 
fit1 <- Arima(staz_1$VALORE, order=c(3,1,1))
summary(fit1)

fit2 <- Arima(staz_1$VALORE, order=c(3,0,2))
summary(fit2) #better error than auto arima but more unstable

fit3 <- auto.arima(staz_1$VALORE, stepwise=FALSE,approximation=FALSE)
summary(fit3)


x11()

matplot(cbind(staz_1$VALORE,fitted(fit3)),type='l')


x11()

matplot(cbind(staz_1$VALORE,fitted(fit2)),type='l')

checkresiduals(fit2)
autoplot(forecast(fit2))
autoplot(fit2)
checkresiduals(fit3)
autoplot(forecast(fit3))
autoplot(fit3)



ggtsdisplay(staz_3$VALORE)
ggtsdisplay(diff(staz_3$VALORE))

fit4 <- Arima(diff(staz_3$VALORE), order=c(3,1,1))
summary(fit4)

fit5 <- Arima(diff(staz_3$VALORE), order=c(3,0,2))
summary(fit5)

fit7<- auto.arima(diff(staz_3$VALORE),stepwise=FALSE,approximation = FALSE)
summary(fit7) #looks like best model

fit6 <- Arima(diff(staz_3$VALORE), order=c(3,0,1))
summary(fit6)

checkresiduals(fit7)
autoplot(forecast(fit7))
autoplot(fit7)










library(bmstdr)

#boxplots area

ptime <- ggplot(data=data_2018,  aes(x=area, y=VALORE)) +
  geom_boxplot() +
  labs(x = "area", y = "PM10 values")  +
  stat_summary(fun=median, geom="line", aes(group=1, col="red")) +
  theme(legend.position = "none")
ptime


# pall <- ggplot(data=data_2018,  aes(x=date, y=VALORE, color=stazione)) +
#   geom_line() +
#   theme(legend.position = "none") +
#   labs(title="PM10 values in ER", y = "values") 
# pall

#head(nyspatial)

library(ggplot2)
library(akima)
library(tidyr)
library(RColorBrewer)
library(ggpubr)
library(spdep)
library(GGally)
library(geoR)
library(fields)
library(doBy)

# head(nysptime)
# 
# a <- nysptime 
# a$time <- rep(1:62, 28)
# a$s.index <- as.factor(a$s.index)
# head(a)



sp <- ggplot(data=data_2018, aes(x=date, y=VALORE, group=stazione, col=stazione)) +
  geom_line() +
  theme(legend.position="none")
sp  #not very useful

p <- ggplot(data=na.omit(data_2018), aes(x=stazione, y=VALORE)) + 
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=2) + 
  labs(title= "station boxplots", x="Station", y = "PM10", size=2.5) 
p  #3 stations seem to have much lower values

s <- c(levels(stazione)[6],levels(stazione)[10],levels(stazione)[13])


# a <- nysptime 
# a$time <- rep(1:62, 28)
# a$s.index <- as.factor(a$s.index)


vdat <- data_2018[which(stazione%in%s), ]
st<-factor(vdat$Stazione)
 ggplot(data=vdat, aes(x=date, y=VALORE,shape=st,color=st)) +
  geom_point() + 
  geom_line() + 
   #facet_wrap(~ stazione, ncol=4) +
  labs(x="Day", y = "PM10", size=2.5) #not very useful



data_2018$Month <- as.factor(data_2018$Month)


#incidence of weekends on PM values




 ggplot(staz_1, aes(x=date, y=VALORE, color=as.factor(weekend))) +
  geom_point(shape=16, size=1.5) +
  stat_smooth(method = "lm", col = "black")+
  labs(x = "days", y = "PM10")
  # geom_text(aes(label=s.index), hjust = -0.7, show.legend = F)+
 




 ggplot(staz_2, aes(x=date, y=VALORE, color=as.factor(weekend))) +
  geom_point(shape=16, size=1.5) +
  stat_smooth(method = "lm", col = "black")+
  labs(x = "days", y = "PM10")
# geom_text(aes(label=s.index), hjust = -0.7, show.legend = F)+





 ggplot(staz_3, aes(x=date, y=VALORE, color=as.factor(weekend))) +
  geom_point(shape=16, size=1.5) +
  stat_smooth(method = "lm", col = "black")+
  labs(x = "days", y = "PM10")
# geom_text(aes(label=s.index), hjust = -0.7, show.legend = F)+
 
 #3 stations that are different by looking at boxplots
 
 staz_6<-data_2018[which(stazione==levels(stazione)[6]),] #CASTELLUCCIO rural/appennino
 
 
 
 
 ndiffs(staz_6$VALORE)
 ggtsdisplay(staz_6$VALORE)
 ggtsdisplay(diff(staz_6$VALORE))
 
 fit8 <- Arima(diff(staz_6$VALORE), order=c(3,0,1))
 summary(fit8)
 
 fit9 <- Arima(diff(staz_6$VALORE), order=c(2,1,1))
 summary(fit9)
 
 fit10<- auto.arima(diff(staz_6$VALORE),stepwise=FALSE,approximation = FALSE)
 summary(fit10)
 

 
 ggplot(staz_6, aes(x=date, y=VALORE, color=as.factor(weekend))) +
   geom_point(shape=16, size=1.5) +
   stat_smooth(method = "lm", col = "black")+
   labs(x = "days", y = "PM10")
 # geom_text(aes(label=s.index), hjust = -0.7, show.legend = F)+
 
  ggplot(data=na.omit(data_2018), aes(x=weekend, y=VALORE)) + 
   geom_boxplot(outlier.colour="red", outlier.shape=8,
                outlier.size=2) + 
   labs(title= "weekend boxplots", x="weekend", y = "PM10", size=2.5) 
  
  
  ggplot(data=na.omit(staz_3), aes(x=weekend, y=VALORE)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=8,
                 outlier.size=2) + 
    labs(title= "weekend boxplots", x="weekend", y = "PM10", size=2.5) 
  
  
  
  
  staz_10<-data_2018[which(stazione==levels(stazione)[10]),] #corte brugnatella rural/pianura est
  ggplot(staz_10, aes(x=date, y=VALORE, color=as.factor(weekend))) +
    geom_point(shape=16, size=1.5) +
    stat_smooth(method = "lm", col = "black")+
    labs(x = "days", y = "PM10")
  
  ggplot(data=na.omit(staz_10), aes(x=weekend, y=VALORE)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=8,
                 outlier.size=2) + 
    labs(title= "weekend boxplots", x="weekend", y = "PM10", size=2.5) 
  
  
  staz_13<-data_2018[which(stazione==levels(stazione)[13]),] #febbio rural/appennino
  
  
  ggplot() + 
    geom_line(data = staz_6, aes(x = date, y = VALORE), color = "red") +
    geom_line(data = staz_13, aes(x = date, y = VALORE), color = "blue") +
    xlab('days') +
    ylab('PM10')
  
  
  
  
  
  ggplot() + 
    geom_line(data = staz_1, aes(x = date, y = VALORE), color = "red") +
    geom_line(data = staz_3, aes(x = date, y = VALORE), color = "blue") +
    xlab('days') +
    ylab('PM10')
  
  ggplot(data=na.omit(urban_data), aes(x=weekend, y=VALORE)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=8,
                 outlier.size=2) + 
    labs(title= "weekend urban data", x="weekend", y = "PM10", size=2.5) 
  
  
  
  ggplot(data=na.omit(suburban_data), aes(x=weekend, y=VALORE)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=8,
                 outlier.size=2) + 
    labs(title= "weekend suburban data", x="weekend", y = "PM10", size=2.5) 
  
  
  ggplot(data=na.omit(rural_data), aes(x=weekend, y=VALORE)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=8,
                 outlier.size=2) + 
    labs(title= "weekend rural data", x="weekend", y = "PM10", size=2.5) 
  
  
  
  
  
  # geom_text(aes(label=s.index), hjust = -0.7, show.legend = F)+

# #missing dates
# 
# date_range <- seq(min(staz_1$date), max(staz_1$date), by = 1) 
# date_range[!date_range %in% staz_1$date] 
# 
# #add missing dates
# library(padr)
# 
# padded_df <- pad(staz_1, interval = "day", end_val = staz_1$date[350])
# padded_df <- fill_by_value(padded_df,VALORE, value =NA )
# padded_df
# 
# 
# 
# 
# data_with_missing_times <- full_join(date_range,staz_1$date)
# 
# 
# output[,1]<- data_2018[which(stazione==levels(stazione)[1]),]$VALORE
# 
# 
#  output <- data.frame(output)
#  class(output)









