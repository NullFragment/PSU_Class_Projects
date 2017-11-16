####################################################################
#### Setup
####################################################################
## Import needed libraries and data
require(reshape2)
require(ggplot2)
require(gridExtra)
require(leaps)
require(car)

## Load data from file
initial_bfp_data=read.table('BODY_FAT.TXT',header=TRUE,sep='\t')

## Recalculate body fat percentage
bfp = (495/initial_bfp_data$Density)-450
bfp[bfp<0] = 0
bfp[bfp>100] = 100
bfp = round(bfp,digits=1)
initial_bfp_data$SiriBFperc <- bfp
initial_bfp_data = initial_bfp_data[-41,]

## Create new bfp_data subset
bfp_data = subset(initial_bfp_data, select = c(Density,SiriBFperc,AbdomenC,Weight,Height,NeckC,ChestC,HipC,ThighC,KneeC,AnkleC,BicepsC,ForearmC,WristC))
bfp_data = subset(bfp_data, bfp_data$Height>50 & bfp_data$SiriBFperc < 41 & bfp_data$SiriBFperc > 2 & bfp_data$Weight < 300)

####################################################################
#### Part A - Backward Elimination
####################################################################
lm_backward = lm(SiriBFperc ~ . - Density,data=bfp_data)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15

lm_backward = update(lm_backward, . ~ . - KneeC)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15

lm_backward = update(lm_backward, . ~ . - ThighC)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15

lm_backward = update(lm_backward, . ~ . - Weight)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15

lm_backward = update(lm_backward, . ~ . - AnkleC)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15

lm_backward = update(lm_backward, . ~ . - BicepsC)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15

lm_backward = update(lm_backward, . ~ . - ChestC)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15

lm_backward = update(lm_backward, . ~ . - ForearmC)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15

lm_backward = update(lm_backward, . ~ . - HipC)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15


####################################################################
#### Part B - Forward Selection
####################################################################
lm_forward = lm(SiriBFperc ~ 1,data=bfp_data)
lm_forward = update(lm_forward, . ~ . + AbdomenC)
lm_forward = update(lm_forward, . ~ . + Weight)
lm_forward = update(lm_forward, . ~ . + WristC)
lm_forward = update(lm_forward, . ~ . + Height)

####################################################################
#### Part C - RSS Selection
####################################################################
lm_full = lm(SiriBFperc ~ . - Density,data=bfp_data)
select_rss = regsubsets(SiriBFperc ~ . - Density,method = "exhaustive",nvmax=13,nbest = 1,data = bfp_data)
select_sum = summary(select_rss)
select_sum$which

## Compute RSS, TSS
p_full=13
p=2:p_full
RSS_p=select_sum$rss
totalSS=sum((bfp_data$SiriBFperc-mean(bfp_data$SiriBFperc))^2)

## Compute Adjusted R2
n=nrow(bfp_data)
R2_adj=1-(RSS_p/(n-p))/(totalSS/(n-1))
plot(p,R2_adj,xlab="Number of betas",ylab="Adjusted R-squared")

## Compute Mallow's C
sigma_hat_full=summary(lm_full)$sigma
C_p=RSS_p/(sigma_hat_full^2)+2*p-n
C_p_diff = abs(abs(C_p)-p)
plot(p,C_p,xlab="Number of betas",ylab="Mallow's Cp")
abline(0,1)

## Compute AIC, BIC
aic_p=n*log(RSS_p/n)+2*p
plot(p,aic_p,xlab="Number of betas",ylab="AIC")
bic_p=n*log(RSS_p/n)+p*log(n)
plot(p,bic_p,xlab="Number of betas",ylab="BIC")

## Create Summary
cbind(select_sum$which,R2_adj,C_p_diff,aic_p,bic_p)
lm_rss = lm(SiriBFperc ~ AbdomenC + Height + NeckC + WristC, data=bfp_data)
