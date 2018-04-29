############################################################################################################################################
#### Setup / Important Functions
############################################################################################################################################
## Install and load libraries
ipak <- function(pkg) {
    new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
    if (length(new.pkg))
        install.packages(new.pkg, dependencies = TRUE)
    sapply(pkg, require, character.only = TRUE)
}

packages <-
    c("ggplot2", 
      "ggfortify", 
      "reshape2", 
      "gridExtra", 
      "TSA", 
      "astsa", 
      "orcutt", 
      "nlme")
ipak(packages)

saveplot <- function(data, xaxis, yaxis, file)
{
    plt = autoplot(data) + xlab(xaxis) + ylab(yaxis)
    png(paste("./figures/", file, ".png", sep = ""), width = 712, height = 400)
    grid.arrange(plt)
    dev.off()
}

saveplot_xy <- function(xdata, ydata, xaxis, yaxis, file)
{
    plt = qplot(x=xdata, y=ydata) + geom_line() + xlab(xaxis) + ylab(yaxis)
    png(paste("./figures/", file, ".png", sep = ""), width = 712, height = 400)
    grid.arrange(plt)
    dev.off()
}

saveacf <- function(data, file)
{
    png(paste("./figures/", file, ".png", sep = ""), width = 712, height = 400)
    acf2(data)
    dev.off()
}

## Load Data
sales_data = read.csv(file = "SalesData.csv", header = TRUE, sep = ",")

## Set important variables
seasonal_trend_val = 12



############################################################################################################################################
#### Initial Data Analysis
############################################################################################################################################
# Plot Time Series
sales = ts(sales_data$Sales)
saveplot(sales, 'time', 'sales', 'sales_timeseries') # Linear trend and seasonality. Detrend with linear transformation.
saveacf(sales, "sales_acf") # 1st or 2nd order AR appearance with no transformation. 12 time step spike -> 12 month seasonality

# Examine periodogram
periodogram(sales)
base_power = periodogram(sales)$spec
base_frequencies = periodogram(sales)$freq
base_periods = 1 / base_frequencies
base_periods_yearly = base_periods / 12

saveplot_xy(
    base_periods_yearly, base_power, 
    "Base Periods in Years", 
    "Base Power", 
    "periodic_base")

############################################################################################################################################
#### Calculate Trends
############################################################################################################################################

# Series Fractional Smoothing
sales_trend = filter(sales, filter = c(1 / (seasonal_trend_val * 2), 
                                      rep(1 / seasonal_trend_val, (seasonal_trend_val - 1)), 
                                      1 / (seasonal_trend_val * 2)), sides = 2)

saveplot(sales_trend, "Time", paste("Fractional Smoothing Trend For", seasonal_trend_val, "Months"), 
         paste("sales_fractional_trend", seasonal_trend_val, "months", sep = "_"))


# Series Exponential Smoothing
sales_arima = arima(sales, order = c(0, 1, 1))
sales_exp = sales - sales_arima$residuals

# Series Decompositions
sales_freq = ts(sales_data$Sales, freq = seasonal_trend_val)

# Additive Decomposition Trend
sales_decomp_add = decompose(sales_freq, type = "additive")
saveplot(sales_decomp_add$trend, "Time", paste("Additive Decomposition Trend For", seasonal_trend_val, "Months"), 
         paste("sales_additive_trend", seasonal_trend_val, "months", sep = "_"))


# Multiplicative Decomposition Trend
sales_decomp_mul = decompose(sales_freq, type = "multiplicative")
saveplot(sales_decomp_mul$trend, "Time", paste("Multiplicative Decomposition Trend For", seasonal_trend_val, "Months"), 
    paste("sales_multiplicative_trend", seasonal_trend_val, "months", sep = "_"))


############################################################################################################################################
#### Detrend Time Series
############################################################################################################################################
# Seasonal Differencing
sales_seasonally = diff(sales_diff, seasonal_trend_val)
saveplot(sales_seasonally, "Time", "Sales", paste("sales_timeseries", seasonal_trend_val, "diff", sep = "_"))

# 1st Differencing
sales_diff = diff(sales, 1)
saveplot(sales_diff, "Time", "Sales", paste("sales_timeseries_1st_and", seasonal_trend_val, "diff", sep = "+"))

# Fractional Smoothing Detrend
sales_frac_smooth = sales - sales_trend

# Exponentials Smoothing Detrend
sales_exp_smooth = sales - sales_exp

# Additive Decomposition Detrend
sales_add_detrend = sales_freq - sales_decomp_add$trend

# Multiplicative Decomposition Detrend
sales_mul_detrend = sales_freq - sales_decomp_mul$trend

############################################################################################################################################
#### Detrended Series Analysis
############################################################################################################################################
# Examine differenced periodogram
periodogram(sales_seasonally)
diff_power = periodogram(sales_seasonally)$spec
diff_frequencies = periodogram(sales_seasonally)$freq
diff_periods = 1 / diff_frequencies
diff_periods_yearly = diff_periods / 12

saveplot_xy(diff_periods_yearly, diff_power, paste(seasonal_trend_val, " Month Differenced Periods in Years", sep = ""), "Power", 
         paste("periodic_", seasonal_trend_val, "_month_detrend", sep =""))

# Detrended Series Plots:
dt_1 <- autoplot(sales_frac_smooth) + ggtitle("Fractional Smoothing") + xlab("Time") + ylab("Detrended Value")
dt_2 <- autoplot(sales_exp_smooth) + ggtitle("Exponential Smoothing") + xlab("Time") + ylab("Detrended Value")
dt_3 <- autoplot(sales_add_detrend) + ggtitle("Additive Detrend") + xlab("Time") + ylab("Detrended Value")
dt_4 <- autoplot(sales_mul_detrend) + ggtitle("Multiplicative Detrend") + xlab("Time") + ylab("Detrended Value")

png(paste("./figures/smoothed_plots", seasonal_trend_val, "month", "trend.png", sep="_"), width = 712, height = 400)
grid.arrange(dt_1, dt_2, dt_3, dt_4, ncol = 2)
dev.off()

# ACF & PACF of Additive and Fractionally Smoothed Data
saveacf(sales_add_detrend, paste("sales_additive_detrend", seasonal_trend_val, "months", "acf", sep = "_"))
saveacf(sales_frac_smooth, paste("sales_fractional_detrend", seasonal_trend_val, "months", "acf", sep = "_"))
saveacf(sales_seasonally, paste("sales_seasonal_detrend", seasonal_trend_val, "months", "acf", sep = "_"))

############################################################################################################################################
#### Fit Time Series Models
############################################################################################################################################
fitModels = FALSE
if(fitModels == TRUE)
{
    # For 12 month trend
    if (seasonal_trend_val == 12) {sarima(sales_seasonally, 1, 1, 1, 1, 1, 1, 12)}
    if (seasonal_trend_val == 12) {sarima(sales, 1, 1, 1, 2, 1, 4, 12)}
    sarima.for(sales, 12, 1, 1, 1, 2, 1, 4, 12)
    
    # For 6 month trend
    if (seasonal_trend_val == 6){sarima(sales_add_detrend, 0, 0, 0, 4, 1, 2, 1)}
    if (seasonal_trend_val == 6){sarima(sales, 1, 1, 1, 2, 1, 4, 12)}
    if (seasonal_trend_val == 6){sarima(sales_seasonally, 1, 1, 1, 1, 1, 1, 6)}
    
    # For 3 month trend
    if (seasonal_trend_val == 3){sarima(sales_add_detrend, 0, 1, 2, 2, 1, 3, 1)}
}

