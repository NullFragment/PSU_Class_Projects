####################################################################
#### Setup
####################################################################
## Import needed libraries and data
library(reshape2)
library(ggplot2)
require(gridExtra)
initial_bfp_data=read.table('BODY_FAT.TXT',header=TRUE,sep='\t')

## Recalculate body fat percentage
bfp = (495/initial_bfp_data$Density)-450
bfp[bfp<0] = 0
bfp[bfp>100] = 100
bfp = round(bfp,digits=1)
initial_bfp_data$SiriBFperc <- bfp

## Create new bfp_data subset
bfp_data = subset(initial_bfp_data, select = c(Density,SiriBFperc,AbdomenC,Weight,Height,NeckC,ChestC,HipC,ThighC,KneeC,AnkleC,BicepsC,ForearmC,WristC))
bfp_data = subset(bfp_data, bfp_data$Height>50 & bfp_data$SiriBFperc < 41 & bfp_data$SiriBFperc > 2 & bfp_data$Weight < 300)

####################################################################
#### Part A
####################################################################
lm_abdomen =lm_abdomen = lm(SiriBFperc~AbdomenC, data = bfp_data)
quantiles = quantile(bfp_data$AbdomenC, probs=c(0.10,.25,.50,.75,.90))
estimates = predict(lm_abdomen, data.frame(AbdomenC=quantiles))


####################################################################
#### Part B
####################################################################
confidence = predict(lm_abdomen, data.frame(AbdomenC=quantiles), interval="confidence", level=.95)
prediction = predict(lm_abdomen, data.frame(AbdomenC=quantiles), interval="prediction", level=.95)

abline(lm_abdomen$coefficients[1],lm_abdomen$coefficients[2],col='green2')
for(i in 1:length(quantiles))
{
  lines(c(quantiles[i],quantiles[i]),c(prediction[i,2],prediction[i,3]),lwd=2, lty=2, col="red2")
  lines(c(quantiles[i],quantiles[i]),c(confidence[i,2],confidence[i,3]),lwd=3, col="blue2")
}


####################################################################
#### Part C
####################################################################
lm_full = lm(SiriBFperc~.-Density, data = bfp_data)
shapiro_reduced = shapiro.test(lm_abdomen$residuals)
shapiro_full = shapiro.test(lm_full$residuals)

p1 <- ggplot(bfp_data, aes(x=lm_abdomen$fitted.values, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p2 <- ggplot(bfp_data, aes(x=lm_full$fitted.values, y=lm_full$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")

qqnorm(lm_abdomen$residuals,main='Reduced Model Q-Q plot')
qqline(lm_abdomen$residuals)

qqnorm(lm_full$residuals,main='Full Model Q-Q plot')
qqline(lm_full$residuals)

####################################################################
#### Display Results
####################################################################

## Part A
quantiles
estimates

## Part B
plot(bfp_data$AbdomenC,bfp_data$SiriBFperc)

## Part C
grid.arrange(p1, p2, ncol=1)

shapiro_reduced
shapiro_full
