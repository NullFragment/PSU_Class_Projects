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
bfp_data = subset(initial_bfp_data, select = c(Density,SiriBFperc,AbdomenC,Weight,Height,NeckC,ChestC,HipC,ThighC,KneeC,AnkleC,BicepsC,ForearmC,WristC,Over45))
bfp_data = subset(bfp_data, bfp_data$Height>50 & bfp_data$SiriBFperc < 41 & bfp_data$SiriBFperc > 2 & bfp_data$Weight < 300)

####################################################################
#### Part A - Backward Elimination
####################################################################
lm_backward = lm(SiriBFperc ~ . - Density,data=bfp_data)
scrap = sort(summary(lm_backward)$coefficients[,4], decreasing=TRUE)
scrap > 0.15

lm_backward = update(lm_backward, . ~ . - Over45)
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
p_full=14
p=2:p_full
RSS_p=select_sum$rss
totalSS=sum((bfp_data$SiriBFperc-mean(bfp_data$SiriBFperc))^2)

## Compute Adjusted R2
n=nrow(bfp_data)
R2_adj=1-(RSS_p/(n-p))/(totalSS/(n-1))
which.max(R2_adj)+1
plot(p,R2_adj,xlab="Number of betas",ylab="Adjusted R-squared")

## Compute Mallow's C
sigma_hat_full=summary(lm_full)$sigma
C_p=RSS_p/(sigma_hat_full^2)+2*p-n
C_p_diff = abs(abs(C_p)-p)
which.min(C_p_diff[1:12])+1
plot(p,C_p,xlab="Number of betas",ylab="Mallow's Cp")
abline(0,1)

## Compute AIC, BIC
aic_p=n*log(RSS_p/n)+2*p
plot(p,aic_p,xlab="Number of betas",ylab="AIC")
which.min(aic_p)+1

bic_p=n*log(RSS_p/n)+p*log(n)
plot(p,bic_p,xlab="Number of betas",ylab="BIC")
which.min(bic_p)+1

## Create Summary
cbind(select_sum$which,R2_adj,C_p_diff,aic_p,bic_p)
lm_rss = lm(SiriBFperc ~ AbdomenC + Height + NeckC + ChestC + HipC + ForearmC + WristC, data=bfp_data)
lm_mallow = lm(SiriBFperc ~ AbdomenC + Weight + WristC, data=bfp_data)
lm_aic = lm(SiriBFperc ~ AbdomenC + Height + NeckC + WristC, data=bfp_data)
lm_bic = lm(SiriBFperc ~ AbdomenC + Weight + WristC, data=bfp_data)


####################################################################
#### Part D - Model Evaluation
####################################################################
## lm_backward = lm_aic
## lm_mallow = lm_bic
vif(lm_backward)
vif(lm_forward)
vif(lm_rss)
vif(lm_mallow)
lm_forward = update(lm_forward, . ~ . - Weight) 
lm_rss = update(lm_rss, . ~ . - AbdomenC) 

## Remove lm_rss from the running. Low R^2 values
adj_r2_sum = c(summary(lm_backward)$adj.r.squared,summary(lm_forward)$adj.r.squared,summary(lm_rss)$adj.r.squared,summary(lm_mallow)$adj.r.squared)
names(adj_r2_sum) <- c("lm_bwd", "lm_fwd", "lm_rss", "lm_bic")

## Remove lm_bic from the running because it has high VIF values compared to fwd and bwd selection.
## Shapiro-Wilk
shapiro_bwd = shapiro.test(lm_backward$residuals)
shapiro_fwd = shapiro.test(lm_forward$residuals)

## Res vs Fitted Vals
p1 <- ggplot(bfp_data, aes(x=lm_backward$fitted.values, y=lm_backward$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p2 <- ggplot(bfp_data, aes(x=lm_forward$fitted.values, y=lm_forward$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
grid.arrange(p1, p2, ncol=1)


## Q-Q
qqnorm(lm_backward$residuals,main='Backward Elimination Model Q-Q plot')
qqline(lm_backward$residuals)

qqnorm(lm_forward$residuals,main='Forward Selection Model Q-Q plot')
qqline(lm_forward$residuals)

## R-plots
plot(lm_backward)
plot(lm_backward,which=c(4,6))
plot(lm_forward)
plot(lm_backward,which=c(4,6))

means = data.frame(rbind(colMeans(bfp_data)))
predict(lm_backward,means, interval="predict")
predict(lm_forward,means, interval="predict")