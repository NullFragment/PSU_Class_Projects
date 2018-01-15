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

packages <- c("ggplot2", "reshape2", "gridExtra", "TSA")
ipak(packages)

####################################################################
#### Problem 2
####################################################################
traffic = read.csv("internet.traffic.csv")
traffic_ts = as.ts(traffic[,2])
plot(traffic_ts, main="Internet Traffic Time Series")
# Traffic dips significantly around christmas & new year
# Peaks seem to occur between 10AM - 3PM

####################################################################
#### Problem 5
####################################################################

## Part A
idx = 1:100
n1 = rnorm(100, mean = 0, sd = 4)
n2 = rnorm(100, mean = 0, sd = 4)
n3 = rnorm(100, mean = 0, sd = 4)
n4 = rnorm(100, mean = 0, sd = 4)
norm = data.frame(idx, n1,n2,n3,n4)

norm_plot = ggplot(norm, aes(idx)) + 
  geom_line(aes(y = n1, colour = "n1")) + 
  geom_line(aes(y = n2, colour = "n2")) + 
  geom_line(aes(y = n3, colour = "n3")) + 
  geom_line(aes(y = n4, colour = "n4")) +
  ggtitle("Normal Distribution") +
  xlab("step") +
  ylab("value")

## Part B
w1 = cumsum(n1)
w2 = cumsum(n2)
w3 = cumsum(n3)
w4 = cumsum(n4)
walk = data.frame(idx, w1,w2,w3,w4)

walk_plot = ggplot(walk, aes(idx)) + 
  geom_line(aes(y = w1, colour = "w1")) + 
  geom_line(aes(y = w2, colour = "w2")) + 
  geom_line(aes(y = w3, colour = "w3")) + 
  geom_line(aes(y = w4, colour = "w4")) +
  ggtitle("Random Walk") +
  xlab("step") +
  ylab("value")

grid.arrange(norm_plot, walk_plot)
