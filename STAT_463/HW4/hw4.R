####################################################################
#### Setup
####################################################################
## Install and load libraries
ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa")
ipak(packages)


## Read data
sim1 = read.csv("sim1hw4.csv", header=FALSE)
huron = read.csv("lakehuron.csv", header=TRUE)
flour = read.table("flourbuffalo.dat", header=TRUE)


####################################################################
#### Problem 2
####################################################################

## Part A
series1 = ts(sim1)
plot(series1, type="b")

## Part B
acf2(series1, 20)

## Part C
sarima(series1, 0, 0, 1)

## Part D
sarima(series1, 0, 0, 2)

## Part G
sarima(series1, 1, 0, 1)

## Part I
sarima.for(series1,6, 0,0,2)


####################################################################
#### Problem 3
####################################################################

## Part A
series2 = ts(huron$Level)
plot(series2, type="b")

## Part B
acf2(series2, 20)

## Part C
sarima(series2, 2, 0, 0)

## Part E


## Part F
sarima.for(series2,3, 2,0,0)



####################################################################
#### Problem 4
####################################################################

## Part A
series3 = ts(flour$Buffalo)
plot(series3, type="b")

## Part B
acf2(series3, 20)

## Part C
diff1=diff(series3,1)
acf2(diff1)
