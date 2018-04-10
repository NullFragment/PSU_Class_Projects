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

packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa", "orcutt")
ipak(packages)


####################################################################
#### Problem 1
####################################################################
fatal_data = read.csv(file="fatals.csv", header=TRUE, sep=",")
travel = fatal_data[,1]
fatals = fatal_data[,2]

## Part A
png("./figures/fatals_scatter.png", width = 1024, height = 576)
plot(fatal_data)
dev.off()


fatal_lm = lm(fatals ~ travel)
# Intercept: 16.75850
# Intercept SE: 2.69288
# Slope: 0.07027
# Slope SE: 0.01754

## Part B
fatal_residuals = fatal_lm$residuals
png("./figures/fatal_lm_residual_acf.png", width = 1024, height = 576)
acf2(fatal_residuals)
dev.off()

# Possible ARIMA: (1,0,0) or (1,0,1)

## Part C

png("./figures/arima_residual_fit.png", width = 1024, height = 576)
sarima(fatal_residuals, 1,0,0, no.constant = TRUE)
dev.off()
# Yes this model is good. QQ and ACF all look god. Only one significant spike in ACF

## Part D
fatals_lag = ts.intersect(fatals, lag(fatals,-1))
fatals_new = fatals_lag[,1] - 0.2944 * fatals_lag[,2]

travel_lag = ts.intersect(travel, lag(travel,-1))
travel_new = travel_lag[,1] - 0.2944 * travel_lag[,2]

adjusted_regression = lm(fatals_new ~ travel_new)
summary(adjusted_regression)


png("./figures/residual_acf.png", width = 1024, height = 576)
acf2(residuals(adjusted_regression))
dev.off()


# Intercept: 54.4500
# Intercept SE: 13.2512
# Slope: 2.7183
# Slope SE: 0.6784

## Part E
summary(cochrane.orcutt(fatal_lm))
co_slope_se = 0.85958
co_slope = 2.416954
co_intercept = 86.082467

#   (Intercept)        travel  
#     86.082467      2.416954 
  
####################################################################
#### Problem 2
####################################################################
fatal_diff = diff(fatals)
travel_diff = diff(travel)

lm_diffs = lm(fatal_diff ~ travel_diff)
png("./figures/p2_lm_diff_acf", width = 1024, height = 576)
acf2(residuals(lm_diffs))
dev.off()

lm_diffs_no_int = lm(fatal_diff ~ 0 + travel_diff)

fatal_mean = mean(fatals)
travel_mean = mean(travel)

diff_slope = 1.936
diff_slope_se = 1.402
diff_int = travel_mean - diff_slope * fatal_mean

####################################################################
#### Problem 3
####################################################################
expected_71 = 34.80

co_predicted_70 = co_intercept + co_slope * travel[70]
co_residual = travel[70] - co_predicted_70
co_predict_71 = co_intercept + co_slope * expected_71
co_forecast = co_predict_71 + co_slope_se * co_residual


diff_predicted_70 = diff_int + diff_slope * travel[70]
diff_residual = travel[70] - diff_predicted_70
diff_predict_71 = diff_int + diff_slope * expected_71
diff_forecast = diff_predict_71 + diff_slope_se * diff_residual
