## Import needed libraries and data
library(reshape2)
library(ggplot2)
require(gridExtra)

swiss=read.table('swiss.txt',header=TRUE,sep='\t')

## Linear System
lm_full = lm(Fertility~., data=swiss)
h = lm.influence(lm_full)$hat
plot(h)

p = length(lm_full$coefficients)
n = length(swiss[,1])
thresh2=2*p/n
thresh3=3*p/n
plot(h,xlab='Observation #',ylab='Leverage',main='Leverage')
abline(h=thresh2,lty=2,col="red")
abline(h=thresh3,lty=2,col="blue")
which(h>thresh2)

r=lm_full$residuals/(summary(lm_full)$sigma * sqrt(1-h))
t=r*sqrt((n-p-1)/(n-p-r^2))
plot(t,xlab='Observation #',ylab='Studentized residuals',main='Studentized residuals')
which(t > 2)
which(t < -2)

x=model.matrix(lm_full, )
D1=(1/p)*r^2*h/(1-h)
plot(x,D1,ylab='Cook\'s distance',main='Cook\'s distance')
plot(D1,xlab='Observation #',ylab='Cook\'s distance',main='Cook\'s distance')
which(D1 >0.1)
