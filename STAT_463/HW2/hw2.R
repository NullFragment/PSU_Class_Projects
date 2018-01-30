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


## Read data
erie = read.csv("eriedata.csv")
oil = read.table("oildata.dat", header=TRUE)
profits = read.table("JandJ.txt", header=TRUE)

####################################################################
#### Problem 1
####################################################################

## Part A
erie_ts = ts (erie[, 2])
plot (erie_ts, type="b")

## Part B
## 14 possible outlier? Seasonality cannot exist due to the data being collected yearly.
## After year 20 the mean appears to have moved to a higher value.

## Part C
erie_acf = acf(erie_ts)
erie_acf

erie_acf_theo_vals = erie_acf$acf[1]^(1:7)
erie_acf_diff = abs(erie_acf$acf[1:7] - erie_acf_theo_vals)
round(erie_acf_theo_vals, 3)
round(erie_acf_diff, 3)
round((erie_acf_diff/erie_acf$acf[1:7])*100, 3)


## For the first 5 or so lag periods it appears to be almost AR(1), however after that, the ACF values deviate significantly

## Part D
lag_erie_ts = lag (erie_ts, -1)
y = cbind (erie_ts, lag_erie_ts)
erie_ar1_model = lm(y[,1] ~ y[,2])
summary (erie_ar1_model)

## Part E
plot(fitted(erie_ar1_model),residuals(erie_ar1_model))

## Part F
acf(residuals(erie_ar1_model))

## Part G
df = data.frame(x = 41)
predict.lm(erie_ar1_model, df)

####################################################################
#### Problem 2
####################################################################

## Part A
oil_ts = ts (oil[, 2])
plot (oil_ts, type="b")

## Part B
oil_ts_diff = diff(oil_ts, 1)
plot (oil_ts_diff, type="b")

## Part C
oil_diff_acf = acf(oil_ts_diff)
oil_diff_acf


####################################################################
#### Problem 3
####################################################################

## Part A
jnj_ts = ts (profits[, 2])
plot (jnj_ts, type="b")

## Part C
log_jnj_ts = log(jnj_ts)
plot (log_jnj_ts, type="b")