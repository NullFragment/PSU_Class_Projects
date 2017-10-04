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
## Fit Linear Models
lm_weight = lm(SiriBFperc~Weight, data = bfp_data)
lm_height = lm(SiriBFperc~Height, data = bfp_data)
lm_abdomen = lm(SiriBFperc~AbdomenC, data = bfp_data)

lm_weight_conf = confint(lm_weight, level = 0.99)
lm_height_conf = confint(lm_height, level = 0.99)
lm_abdomen_conf = confint(lm_abdomen, level = 0.99)

####################################################################
#### Part B
####################################################################
t_abdomen = (summary(lm_abdomen)$coefficients[2] - 0.5)/summary(lm_abdomen)$coefficients[4]
p_abdomen = 1-pnorm(t_abdomen)

####################################################################
#### Part C
####################################################################

p1 <- ggplot(bfp_data, aes(x=bfp_data$AbdomenC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p2 <- ggplot(bfp_data, aes(x=bfp_data$Height, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p3 <- ggplot(bfp_data, aes(x=bfp_data$NeckC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p4 <- ggplot(bfp_data, aes(x=bfp_data$ChestC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p5 <- ggplot(bfp_data, aes(x=bfp_data$HipC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p6 <- ggplot(bfp_data, aes(x=bfp_data$ThighC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p7 <- ggplot(bfp_data, aes(x=bfp_data$KneeC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p8 <- ggplot(bfp_data, aes(x=bfp_data$AnkleC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p9 <- ggplot(bfp_data, aes(x=bfp_data$BicepsC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p10 <- ggplot(bfp_data,aes(x=bfp_data$ForearmC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")
p11 <- ggplot(bfp_data,aes(x=bfp_data$WristC, y=lm_abdomen$residuals)) + geom_point(size=1) + geom_hline(yintercept = 0, color="red")

####################################################################
#### Part D
####################################################################
# Create full linear model
lm_full = lm(SiriBFperc~.-Density, data = bfp_data)

# Calculate f-statistic and p-value comparing full model and only abdomen circumference
RSS_full = sum(lm_full$residuals^2)
RSS_reduced = sum(lm_abdomen$residuals^2)
n = dim(bfp_data)[1]
p=length(lm_full$coefficients)
q=length(lm_abdomen$coefficients)
f=((RSS_reduced-RSS_full)/(p-q))/(RSS_full/(n-p))
pvalue=pf(f,p-q,n-p,lower.tail=FALSE)

####################################################################
#### Display Results
####################################################################

## Part A
summary(lm_weight)
summary(lm_height)
summary(lm_abdomen)

lm_weight_conf
lm_height_conf
lm_abdomen_conf

# Part B
t_abdomen
p_abdomen

# Part C
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, ncol=3)


# Part D
f
pvalue

# Check results
anova(lm_abdomen,lm_full)


