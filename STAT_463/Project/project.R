####################################################################
#### Setup / Important Functions
####################################################################
## Install and load libraries
ipak <- function(pkg) {
    new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
    if (length(new.pkg)) 
        install.packages(new.pkg, dependencies = TRUE)
    sapply(pkg, require, character.only = TRUE)
}

packages <- c("ggplot2", "ggfortify", "reshape2", "gridExtra", "TSA", "astsa", "orcutt", "nlme")
ipak(packages)

# png("./figures/interaction.png", width = 1024, height = 576)
# Plot Statment
# dev.off()

####################################################################
#### DATA ANALYSIS
####################################################################
## Set important variables 
seasonal_trend_val = 12

## Load and Plot Data
sales_data = read.csv(file="SalesData.csv", header=TRUE, sep=",")
sales = ts(sales_data$Sales)
plot(sales) # Linear trend and seasonality. Detrend with linear transformation.
acf2(sales) # 1st or 2nd order AR appearance with no transformation. 12 time step spike -> 12 month seasonality

# Examine periodogram
periodogram(sales)
base_power = periodogram(sales)$spec
base_frequencies = periodogram(sales)$freq
base_periods = 1/base_frequencies
base_periods_yearly = base_periods/12
plot(x=base_periods_yearly, y=base_power)

# 1st and 12th Differencing
sales_diff = diff(sales, 1)
plot(sales_diff)
sales_yearly = diff(sales_diff, seasonal_trend_val)
plot(sales_yearly) # Still noisy. Sinusoidal

# Examine differenced periodogram
periodogram(sales_yearly)
diff_power = periodogram(sales_yearly)$spec
diff_frequencies = periodogram(sales_yearly)$freq
diff_periods = 1/diff_frequencies
diff_periods_yearly = diff_periods/12
plot(x=diff_periods_yearly, y=diff_power)


# Series Fractional Smoothing
sales_trend = filter(sales, filter=c(1/(seasonal_trend_val*2), rep(1/seasonal_trend_val, (seasonal_trend_val*2 - 1)), 1/(seasonal_trend_val*2)), sides=2)
plot(sales_trend)
sales_frac_smooth = sales - sales_trend

# Series Exponential Smoothing
sales_arima = arima(sales, order = c(0,1,1))
sales_exp = sales-sales_arima$residuals
sales_exp_smooth = sales - sales_exp

# Series Decompositions
sales_freq = ts(sales_data$Sales, freq=seasonal_trend_val)
sales_decomp_add = decompose(sales_freq, type="additive")
sales_decomp_mul = decompose(sales_freq, type="multiplicative")

# Additive Decomposition Detrend
plot(sales_decomp_add)
sales_add_detrend = sales_freq - sales_decomp_add$trend

# Multiplicative Decomposition Detrend
plot(sales_decomp_mul)
sales_mul_detrend = sales_freq - sales_decomp_mul$trend

# Lowess Smoothing
sales_lowess = stl(sales_freq, "periodic")
sales_lowess_detrend = sales_freq - sales_lowess$time.series[,2]

####################################################################
#### Detrended Series Analysis
####################################################################
# Detrended Series Plots:
dt_1 <- autoplot(sales_frac_smooth) + ggtitle("Fractional Smoothing")
dt_2 <- autoplot(sales_exp_smooth) + ggtitle("Exponential Smoothing")
dt_3 <- autoplot(sales_add_detrend) + ggtitle("Additive Detrend")
dt_4 <- autoplot(sales_mul_detrend) + ggtitle("Multiplicative Detrend")
grid.arrange(dt_1, dt_2, dt_3, dt_4, ncol = 2)

# Fractional Smoothing
fs_1 <- autoplot(sales_add_detrend) + ggtitle("Additive Detrend")
fs_2 <- autoplot(diff(sales_add_detrend,1)) + ggtitle("Additive Detrend w/ 1st Difference")
grid.arrange(fs_1, fs_2, ncol=2)
acf2(sales_add_detrend)

# For 12 month trend
if (seasonal_trend_val == 12){sarima(sales_add_detrend,0,0,0,4,1,2,1)}
if (seasonal_trend_val == 12){sarima(sales,1,1,1,2,1,4,12)}
sarima.for(sales,12,1,1,1,2,1,4,12)

# For 6 month trend
if (seasonal_trend_val == 6){sarima(sales_add_detrend,0,0,0,4,1,2,1)}

# For 3 month trend
if (seasonal_trend_val == 3){sarima(sales_add_detrend,0,1,2,2,1,3,1)}

