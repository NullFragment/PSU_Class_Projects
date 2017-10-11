## Import needed libraries and data
library(reshape2)
library(ggplot2)
require(gridExtra)

bear=read.table('bears.txt',header=TRUE,sep='\t')
bear=bear[bear$Obs.No==1,]

## Linear System
lm_length = lm(bear$Weight~bear$Length)
p1 <- ggplot(bear, aes(x=lm_length$fitted.values, y=lm_length$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
plot(p1)
qqnorm(lm_length$residuals,main='Residuals Q-Q plot')
qqline(lm_length$residuals)
shapiro.test(lm_length$residuals)

## Log Transformed System
lm_log = lm(log(bear$Weight)~bear$Length)
p2 <- ggplot(bear, aes(x=lm_log$fitted.values, y=lm_log$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
plot(p2)
qqnorm(lm_log$residuals,main='Residuals Q-Q plot')
qqline(lm_log$residuals)
shapiro.test(lm_log$residuals)



## Polynomial Transformed System
lm_transform = lm(bear$Weight~bear$Length + I((bear$Length)^2))
p3 <- ggplot(bear, aes(x=lm_transform$fitted.values, y=lm_transform$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
plot(p3)
qqnorm(lm_transform$residuals,main='Residuals Q-Q plot')
qqline(lm_transform$residuals)
shapiro.test(lm_transform$residuals)