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

## Part B

## Part C
plot(milk_additive_decomp)

## Part D
milk_multiplicative_decomp = decompose(milk_ts, type="multiplicative")

## Part E

## Part F
milk_lowess_decomp = stl(milk_ts, "periodic")

## Part G
# Yes, the trend is linear, not exponential or polynomial so the additive
# decomposition is suitable.

####################################################################
#### Problem 2
####################################################################

## Part A
acf2(milk)
milk_differenced_data = diff(milk, 1)
milk_differenced_data = diff(milk_differenced_data, 12)
acf2(milk_differenced_data)

# Possibly (1,1,1,2,1,2,12) SARIMA

## Part B
sarima(milk_ts, 1,1,1,2,1,2,12)

## Part C
sarima.for(milk_ts,12, 1,1,1,2,1,2,12)

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
plot(flour_trend)

## Part B
flour_minus_trend_ts = flour_ts - flour_trend_ts
plot(flour_minus_trend_ts)

## Part C
flour_lowess_ts = lowess(flour_ts, f=1/4)
plot(flour_lowess_ts)

## Part D
flour_lowess_ts_10 = lowess(flour_ts, f=1/10)
flour_lowess_ts_5 = lowess(flour_ts, f=1/5)
flour_lowess_ts_1 = lowess(flour_ts, f=1)

plot(flour_lowess_ts_10)
points(flour_lowess_ts_5, col="red")
points(flour_lowess_ts_1, col="green")

# As f increases, the plot approaches a straight line at the mean

## Part E
flour_arima = arima(flour_ts, order = c(0,1,1))
flour_arima_coeff = unname(flour_arima$coef)
flour_arima_sigma = unname(sqrt(flour_arima$sigma2))

## Part F
flour_pred_vals = flour_ts - flour_arima$residuals
plot(flour_ts, type="b")
lines(flour_pred_vals, col="red")

## Part G
flour_alpha = 1 + flour_arima_coeff
flour_alpha*flour_ts[100]+(1 - flour_alpha)*flour_pred_vals[100]