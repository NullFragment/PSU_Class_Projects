####################################################################
#### Setup
####################################################################
## Install and load libraries
# ipak function taken from: https://gist.github.com/stevenworthington/3178163
# ipak <- function(pkg) {
#   new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
#   if (length(new.pkg)) 
#     install.packages(new.pkg, dependencies = TRUE)
#   sapply(pkg, require, character.only = TRUE)
# }
# 
# packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa", "orcutt", 
#               "nlme", "fGarch", "vars")
# ipak(packages)

# Set up variables for first few questions
treatmentsSorted = c(rep("none",5), rep("low", 5), rep("medium", 5), rep("high", 5))
units = 1:length(treatmentsSorted)

####################################################################
#### Problem 1
####################################################################
sample1 = sample(treatmentsSorted)
experiment1 = data.frame(units, sample1)
experiment1

####################################################################
#### Problem 2
####################################################################
sample2 = sample(treatmentsSorted)
experiment2 = data.frame(units, sample2)
experiment2

####################################################################
#### Problem 3
####################################################################
q3treats = c(rep("r1", 3), rep("r2", 5), rep("r3", 5))
q3units = 1:length(q3treats)
q3sample = sample(q3treats)
q3experiment = data.frame(q3units, q3sample)
q3experiment


####################################################################
#### Problem 6
####################################################################
## Part A
q6_X = rnorm(1000, mean = -2, sd = 3)

png("./figures/p6_a.png", width = 1024, height = 576)
hist(q6_X)
dev.off()

## Part B
q6_Y = rnorm(1000, mean = 3, sd = 1)

png("./figures/p6_b.png", width = 1024, height = 576)
hist(q6_Y)
dev.off()

## Part C
q6_Z = q6_X + q6_Y

png("./figures/p6_c.png", width = 1024, height = 576)
hist(q6_Z)
dev.off()


## Part E
# Z ~ N(1, 4)
mean(q6_Z)
sd(q6_Z)
