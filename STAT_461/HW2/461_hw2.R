####################################################################
#### Setup
####################################################################
## Install and load libraries
# ipak function taken from: https://gist.github.com/stevenworthington/3178163
ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa", "orcutt", 
              "nlme", "fGarch", "vars")
ipak(packages)

# Set up variables for first few questions
treatmentsSorted = c(rep("none",5), rep("low", 5), rep("medium", 5), rep("high", 5))
units = 1:length(treatmentsSorted)


####################################################################
#### Problem 1
####################################################################
sample1 = sample(treatmentsSorted)
experiment1 = data.frame(units, sample1)

####################################################################
#### Problem 2
####################################################################
sample2 = sample(treatmentsSorted)
experiment2 = data.frame(units, sample2)

####################################################################
#### Problem 3
####################################################################
q3treats = c(rep("r1", 3), rep("r2", 5), rep("r3", 5))
q3units = 1:length(q3treats)
q3sample = sample(q3treats)
q3experiment = data.frame(q3units, q3sample)

# Question 4
# The first correlation was the lighthearted correlation of:
# US spending on science, space and technology
# Suicides by hanging, strangulation and suffocation
# 
# One factor that could increase suicide rates would be the amount of automation
# implemented by advances in technology, which drives people out of jobs. These
# forms of suicide are also fairly cheap to accomplish and the loss of income
# could push people who are already struggling in life over their breaking point.

# Question 5
# a) W ~ N(2-3+0, 6+2+1) => W ~ N(-1, 9)
# b) Q = 2Y = Y + Y ~ N(-3-3, 2+2) => Q ~ N(-6, 4)
# c) P = -2X + 4; 4 ~ N(4,0); P ~ N(-2-2+4, 6+6,0) => P ~ N(0, 12)
# d) Any real values of a and b could be used. For example: a = 5, b = 1: M ~ N(5+1*2, 6) => M ~ N(7, 6)

####################################################################
#### Problem 6
####################################################################
## Part A
q6_X = rnorm(1000, mean = -2, sd = 3)
hist(q6_X)

## Part B
q6_Y = rnorm(1000, mean = 3, sd = 1)
hist(q6_Y)

## Part C
q6_Z = q6_X + q6_Y
hist(q6_Z)

## Part D
# Yes, although the distribution is a linear combination of the two independent
# distributions, the samples from Z are not dependent on samples from X or Y.
# P(Z|A,B) = P(Z)

## Part E
# Z ~ N(1, 4)
mean(q6_Z)
sd(q6_Z)







####################################################################
#### Problem 1
####################################################################

## Part A
