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
milk=read.table("pamilk.txt", header=T)[,1]
flour = read.delim("flour.dat", header = TRUE, sep = "") [,3]

####################################################################
#### Problem 1
####################################################################

milk_ts = ts(milk, start = c(1, 6), frequency = 12)

## Part A
milk_additive_decomp = decompose(milk_ts, type="additive")
milk_additive_decomp$seasonal

## Part B
735 - 95.226620

## Part C
png("./figures/p1_part_c.png", width = 1024, height = 576)
plot(milk_additive_decomp)
dev.off()

## Part D
milk_multiplicative_decomp = decompose(milk_ts, type="multiplicative")
milk_multiplicative_decomp$seasonal

## Part E

## Part F
milk_lowess_decomp = stl(milk_ts, "periodic")

## Part G
# 

####################################################################
#### Problem 2
####################################################################

## Part A
png("./figures/p2_part_a_1.png", width = 1024, height = 576)d
acf2(milk)
dev.off()

milk_differenced_data = diff(milk, 1)
milk_differenced_data = diff(milk_differenced_data, 12)

png("./figures/p2_part_a_2.png", width = 1024, height = 576)
acf2(milk_differenced_data)
dev.off()

# Possibly (1,1,1,2,1,2,12) SARIMA

## Part B
png("./figures/p2_part_b.png", width = 1024, height = 576)
sarima(milk_ts, 1,1,1,1,1,1,12)
dev.off()

## Part C
png("./figures/p2_part_c.png", width = 1024, height = 576)
sarima.for(milk_ts,12, 1,1,1,1,1,1,12)
dev.off()

####################################################################
#### Problem 3
####################################################################
flour_ts = ts(flour)
plot(flour_ts, type="b")
abline(v=seq(0,100,3), col="blue")
abline(v=seq(0,100,12), col="red")

## Part A

flour_trend = filter(flour_ts, filter=c(1/8, rep(1/4, 3), 1/8), sides=2)
#flour_trend = filter(flour_ts, filter=c(1/24,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/24), sides=2)
#flour_trend = filter(flour_ts, filter=c(1/120, rep(1/60, 59), 1/120), sides=2)

flour_trend_ts = ts(flour_trend)

png("./figures/p3_part_a.png", width = 1024, height = 576)
plot(flour_trend)
dev.off()

## Part B
flour_minus_trend_ts = flour_ts - flour_trend_ts

png("./figures/p3_part_b.png", width = 1024, height = 576)
plot(flour_minus_trend_ts)
dev.off()

## Part C
flour_lowess_ts = lowess(flour_ts, f=1/4)

png("./figures/p3_part_c.png", width = 1024, height = 576)
plot(flour_lowess_ts)
dev.off()

## Part D
flour_lowess_ts_10 = lowess(flour_ts, f=1/10)
flour_lowess_ts_5 = lowess(flour_ts, f=1/5)
flour_lowess_ts_1 = lowess(flour_ts, f=1)

png("./figures/p3_part_d.png", width = 1024, height = 576)
plot(flour_lowess_ts_10)
points(flour_lowess_ts_5, col="red")
points(flour_lowess_ts_1, col="green")
dev.off()

# As f increases, the plot approaches a straight line at the mean

## Part E
flour_arima = arima(flour_ts, order = c(0,1,1))
flour_arima_coeff = unname(flour_arima$coef)
flour_arima_sigma = unname(sqrt(flour_arima$sigma2))

## Part F
flour_pred_vals = flour_ts - flour_arima$residuals

png("./figures/p3_part_f.png", width = 1024, height = 576)
plot(flour_ts, type="b")
lines(flour_pred_vals, col="red")
dev.off()

## Part G
flour_alpha = 1 + flour_arima_coeff
flour_alpha*flour_ts[100]+(1 - flour_alpha)*flour_pred_vals[100]
