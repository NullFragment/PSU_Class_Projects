####################################################################
#### Setup
####################################################################
## Import needed libraries and data
library(reshape2)
library(ggplot2)
initial_bfp_data=read.table('BODY_FAT.TXT',header=TRUE,sep='\t')

####################################################################
#### Part A
####################################################################
## Recalculate body fat percentage
bfp = (495/initial_bfp_data$Density)-450
bfp[bfp<0] = 0
bfp[bfp>100] = 100
bfp = round(bfp,digits=1)
num_wrong = length(bfp) - sum((bfp == initial_bfp_data$SiriBFperc), na.rm=TRUE)
initial_bfp_data$SiriBFperc <- bfp

## Create new bfp_data subset
bfp_data = subset(initial_bfp_data, select = c(Density,SiriBFperc,Weight,Height,AbdomenC))
bfp_data = subset(bfp_data, bfp_data$Height>50 & bfp_data$SiriBFperc < 41 & bfp_data$SiriBFperc > 2 & bfp_data$Weight < 300)


####################################################################
#### Part B
####################################################################
## Create histograms of all bfp_data
ggplot(data=melt(bfp_data),aes(x = value)) + facet_wrap(~variable,scales = "free_x") + geom_histogram()

## Create xy plots of all bfp_data
plot(bfp_data)

## Generate bfp_data summary and std deviations
summary(bfp_data)
round(sapply(bfp_data, sd), digits=2)

## Perform T-Test
## T = (Xbar - mu)/(S/sqrt(N))


t_bfp = (mean(bfp_data$SiriBFperc) - 20)/(sd(bfp_data$SiriBFperc)/sqrt(length(bfp_data$SiriBFperc)))
t_w = (mean(bfp_data$Weight) - 180)/(sd(bfp_data$Weight)/sqrt(length(bfp_data$Weight)))
p_bfp = 1-pnorm(t_bfp)
p_w = 1-pnorm(t_w)


####################################################################
#### Part C
####################################################################

## Fit linear model between density and weight
lin_fit_w = lm(bfp_data$SiriBFperc ~ bfp_data$Weight, data = bfp_data)
summary(lin_fit_w)
lin_fit_w$coefficients
sigma(lin_fit_w)^2

plot(bfp_data$Weight, bfp_data$SiriBFperc)
abline(lin_fit_w$coefficients[1],lin_fit_w$coefficients[2],col='red')

## Fit linear model between density and height
lin_fit_h = lm(bfp_data$SiriBFperc ~ bfp_data$Height, data = bfp_data)
summary(lin_fit_h)
lin_fit_h$coefficients
sigma(lin_fit_h)^2

plot(bfp_data$Height, bfp_data$SiriBFperc)
abline(lin_fit_h$coefficients[1],lin_fit_h$coefficients[2],col='red')

## Fit linear model between density and abdomen circumference
lin_fit_a = lm(bfp_data$SiriBFperc ~ bfp_data$AbdomenC, data = bfp_data)
summary(lin_fit_a)
lin_fit_a$coefficients
sigma(lin_fit_a)^2

plot(bfp_data$AbdomenC, bfp_data$SiriBFperc)
abline(lin_fit_a$coefficients[1],lin_fit_a$coefficients[2],col='red')


####################################################################
#### Part D
####################################################################
weight_height = bfp_data$Weight/bfp_data$Height
lin_fit_np = lm(bfp_data$SiriBFperc ~ weight_height, data = bfp_data)
summary(lin_fit_np)
lin_fit_np$coefficients
sigma(lin_fit_np)^2

plot(weight_height, bfp_data$SiriBFperc)
abline(lin_fit_np$coefficients[1],lin_fit_np$coefficients[2],col='red')


####################################################################
#### Part E
####################################################################

lin_fit_ratio = lm(weight_height ~ bfp_data$AbdomenC, data = bfp_data)
summary(lin_fit_ratio)
lin_fit_ratio$coefficients
sigma(lin_fit_ratio)^2

plot(bfp_data$AbdomenC,weight_height)
abline(lin_fit_ratio$coefficients[1],lin_fit_ratio$coefficients[2],col='red')
