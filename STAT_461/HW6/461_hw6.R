###############################################################################
#### Setup
###############################################################################
## Install and load libraries
# ipak function taken from: https://gist.github.com/stevenworthington/3178163
ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}
packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa", "orcutt",
              "nlme", "fGarch", "vars", "lsmeans")
ipak(packages)

###############################################################################
#### Data Input
###############################################################################
type<-c("AlkName","AlkName","AlkName","AlkName","AlkGen","AlkGen",
        "AlkGen","AlkGen","HDName","HDName","HDName","HDName",
        "HDGen","HDGen","HDGen","HDGen")

life<-c(100.668, 77.734,79.210,95.063,206.880,153.347,165.980,196.000,
        14.951,18.063,11.111,12.840,15.340,22.090,15.734, 14.440)

batt<-data.frame(type=type, life=life)

###############################################################################
#### Problem 1
###############################################################################
png("./figures/battery-boxplot.png", width = 1024, height = 576)
  boxplot(life ~ type, main="Boxplot of Battery Data")
dev.off()

###############################################################################
#### Problem 2
###############################################################################
## Part B
lifeAnova = aov(life~type, data=batt)
summary(lifeAnova)

png("./figures/original-ResidualVsFitted.png", width = 1024, height = 576)
  plot(lifeAnova,1)
dev.off()

png("./figures/original-QQ.png", width = 1024, height = 576)
  plot(lifeAnova,2)
dev.off()

###############################################################################
#### Problem 3
###############################################################################
## Part B
sqrtLifeAnova = aov(sqrt(life)~type, data=batt)
summary(sqrtLifeAnova)

png("./figures/root-ResidualVsFitted.png", width = 1024, height = 576)
  plot(sqrtLifeAnova,1)
dev.off()

png("./figures/root-QQ.png", width = 1024, height = 576)
  plot(sqrtLifeAnova,2)
dev.off()

###############################################################################
#### Problem 4
###############################################################################
## Part B
logLifeAnova = aov(log(life)~type, data=batt)
summary(logLifeAnova)

png("./figures/log-ResidualVsFitted.png", width = 1024, height = 576)
  plot(logLifeAnova,1)
dev.off()

png("./figures/log-QQ.png", width = 1024, height = 576)
  plot(logLifeAnova,2)
dev.off()

###############################################################################
#### Problem 5
###############################################################################
## Part B
squareLifeAnova = aov((life^2)~type, data=batt)
summary(squareLifeAnova)

png("./figures/square-ResidualVsFitted.png", width = 1024, height = 576)
  plot(squareLifeAnova,1)
dev.off()

png("./figures/square-QQ.png", width = 1024, height = 576)
  plot(squareLifeAnova,2)
dev.off()

###############################################################################
#### Problem 7
###############################################################################
sqrtLifeLSM = lsmeans(sqrtLifeAnova, ~ type)
contrast(sqrtLifeLSM, method='pairwise')

###############################################################################
#### Problem 8
###############################################################################
hotdog=read.table("hotdogs.txt",header=TRUE)

png("./figures/hotdog-boxplot.png", width = 1024, height = 576)
  boxplot(hotdog$Calories~hotdog$Type, main="Boxplot of Hotdog Types vs Calories")
dev.off()

hotdogModel = aov(Calories ~ Type, data=hotdog)
summary(hotdogModel)
hotdogLSM = lsmeans(hotdogModel, ~Type)
contrast(hotdogLSM, method='pairwise')
