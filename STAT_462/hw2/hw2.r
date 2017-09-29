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

summary(lm_weight)
summary(lm_height)
summary(lm_abdomen)

confint(lm_weight, level = 0.99)
confint(lm_height, level = 0.99)
confint(lm_abdomen, level = 0.99)

####################################################################
#### Part B
####################################################################

####################################################################
#### Part C
####################################################################
lm_NeckC = lm(SiriBFperc~NeckC, data = bfp_data)
lm_ChestC = lm(SiriBFperc~ChestC, data = bfp_data)
lm_HipC = lm(SiriBFperc~HipC, data = bfp_data)
lm_ThighC = lm(SiriBFperc~ThighC, data = bfp_data)
lm_KneeC = lm(SiriBFperc~KneeC, data = bfp_data)
lm_AnkleC = lm(SiriBFperc~AnkleC, data = bfp_data)
lm_BicepsC = lm(SiriBFperc~BicepsC, data = bfp_data)
lm_ForearmC = lm(SiriBFperc~ForearmC, data = bfp_data)
lm_WristC = lm(SiriBFperc~WristC, data = bfp_data)

p1 <- ggplot(bfp_data, aes(x=lm_weight$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p2 <- ggplot(bfp_data, aes(x=lm_height$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p3 <- ggplot(bfp_data, aes(x=lm_NeckC$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p4 <- ggplot(bfp_data, aes(x=lm_ChestC$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p5 <- ggplot(bfp_data, aes(x=lm_HipC$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p6 <- ggplot(bfp_data, aes(x=lm_ThighC$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p7 <- ggplot(bfp_data, aes(x=lm_KneeC$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p8 <- ggplot(bfp_data, aes(x=lm_AnkleC$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p9 <- ggplot(bfp_data, aes(x=lm_BicepsC$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p10 <- ggplot(bfp_data, aes(x=lm_ForearmC$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)
p11 <- ggplot(bfp_data, aes(x=lm_WristC$residuals, y=lm_abdomen$residuals)) + geom_point(size=1)


grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, ncol=3)
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

# Check results
anova(lm_abdomen,lm_full)


y=c(3,-2,7,4)
x1=c(5,5,-5,-5)
x2=c(2,0,2,0)
X=model.matrix(y~x1+x2)
a = solve(t(X)%*%X)
b = t(X)%*%y
a%*%b
1-0.5*x1+2*x2
