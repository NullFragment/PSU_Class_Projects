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
lin_fit_w = lm(bfp_data$Density ~ bfp_data$Weight, data = bfp_data)
summary(lin_fit_w)

## Fit linear model between density and height
lin_fit_h = lm(bfp_data$Density ~ bfp_data$Height, data = bfp_data)
summary(lin_fit_h)

## Fit linear model between density and abdomen circumference
lin_fit_a = lm(bfp_data$Density ~ bfp_data$AbdomenC, data = bfp_data)
summary(lin_fit_a)


####################################################################
#### Part
####################################################################
####################################################################
#### Part
####################################################################
####################################################################
#### Part
####################################################################

