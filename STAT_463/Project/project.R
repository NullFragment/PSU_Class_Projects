#####################################################################################
#### Setup / Important Functions
#####################################################################################
## Install and load libraries
ipak <- function(pkg) {
    new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
    if (length(new.pkg))
        install.packages(new.pkg, dependencies = TRUE)
    sapply(pkg, require, character.only = TRUE)
}

packages <-
    c("ggplot2", "ggfortify", "reshape2", "gridExtra", "TSA", "astsa", "orcutt", "nlme")
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

#####################################################################################
#### Define Important Variables
#####################################################################################
## Load Data
sales_data = read.csv(file = "SalesData.csv", header = TRUE, sep = ",")

## Set seasonality value
seasonal_trend_val = 12
fitModels = FALSE
forecastModels = TRUE


#####################################################################################
#### Initial Data Analysis
#####################################################################################
# Plot Time Series
sales = ts(sales_data$Sales)
saveplot(sales, 'time', 'sales', 'time_series_sales') # Linear trend and seasonality. Detrend with linear transformation.
saveacf(sales, "acf_sales") # 1st or 2nd order AR appearance with no transformation. 12 time step spike -> 12 month seasonality

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


#####################################################################################
#### Calculate Trends
#####################################################################################

# Series Fractional Smoothing
sales_trend = filter(sales, filter = c(1 / (seasonal_trend_val * 2), 
                                       rep(1 / seasonal_trend_val, (seasonal_trend_val - 1)), 
                                       1 / (seasonal_trend_val * 2)), sides = 2)

# Series Exponential Smoothing
sales_arima = arima(sales, order = c(0, 1, 1))
sales_exp = sales - sales_arima$residuals

# Series Decompositions
sales_freq = ts(sales_data$Sales, freq = seasonal_trend_val)

# Additive Decomposition Trend
sales_decomp_add = decompose(sales_freq, type = "additive")

# Multiplicative Decomposition Trend
sales_decomp_mul = decompose(sales_freq, type = "multiplicative")

# Set Plot Titles
fractional_title = paste("Fractional Smoothing Trend For", seasonal_trend_val, "Months")
exponential_title = paste("Exponential Smoothing Trend For", seasonal_trend_val, "Months")
additive_title = paste("Additive Decomposition Trend For", seasonal_trend_val, "Months")
multiplicative_title = paste("Multiplicative Decomposition Trend For", seasonal_trend_val, "Months")

# Save Trends Plot
trend_1 <- autoplot(sales_trend) + xlab("Time") + ylab("Fractional Smoothing Trend Values") + ggtitle(fractional_title)
trend_2 <- autoplot(sales_exp) + xlab("Time") + ylab("Exponential Smoothing Trend Values") + ggtitle(exponential_title)
trend_3 <- autoplot(sales_decomp_add$trend) + xlab("Time") + ylab("Additive Decomposition Trend Values") + ggtitle(additive_title)
trend_4 <- autoplot(sales_decomp_mul$trend) + xlab("Time") + ylab("Multiplicative Decomposition Trend Values") + ggtitle(multiplicative_title)

png(paste("./figures/trends_for", seasonal_trend_val, "month", "season.png", sep="_"), width = 712, height = 400)
grid.arrange(trend_1, trend_2, trend_3, trend_4, ncol = 2)
dev.off()

#####################################################################################
#### Basic Differencing
#####################################################################################
# Seasonal Difference
sales_seasonally = diff(sales, seasonal_trend_val)

# 1st Difference
sales_diff = diff(sales_seasonally, 1)

# Set Plot Titles
seasonal_title = paste(seasonal_trend_val, "Month Differenced Time Series")
seasonal_and_1st_title = paste(seasonal_trend_val, "Month and 1st Differenced Time Series")

# Differencing Plots
diff_1 <- autoplot(sales_seasonally) + xlab("Time") + ylab("Sales") + ggtitle(seasonal_title)
diff_2 <- autoplot(sales_diff) + xlab("Time") + ylab("Sales") + ggtitle(seasonal_and_1st_title)

png(paste("./figures/differencing_for", seasonal_trend_val, "month", "season.png", sep="_"), width = 712, height = 400)
grid.arrange(diff_1, diff_2, ncol = 1)
dev.off()

#####################################################################################
#### Detrend Time Series
#####################################################################################
# Fractional Smoothing Detrend
sales_frac_smooth = sales - sales_trend

# Exponentials Smoothing Detrend
sales_exp_smooth = sales - sales_exp

# Additive Decomposition Detrend
sales_add_detrend = sales_freq - sales_decomp_add$trend

# Multiplicative Decomposition Detrend
sales_mul_detrend = sales_freq - sales_decomp_mul$trend


#####################################################################################
#### Detrended Series Analysis
#####################################################################################
# Examine differenced periodogram
periodogram(sales_seasonally)
diff_power = periodogram(sales_seasonally)$spec
diff_frequencies = periodogram(sales_seasonally)$freq
diff_periods = 1 / diff_frequencies
diff_periods_yearly = diff_periods / 12

saveplot_xy(diff_periods_yearly, diff_power, paste(seasonal_trend_val, " Month Differenced Periods in Years", sep = ""), "Power", 
            paste("periodic", seasonal_trend_val, "month_detrend", sep ="_"))

# Detrended Series Plots:
dt_1 <- autoplot(sales_frac_smooth) + ggtitle("Fractional Smoothing") + xlab("Time") + ylab("Detrended Value")
dt_2 <- autoplot(sales_exp_smooth) + ggtitle("Exponential Smoothing") + xlab("Time") + ylab("Detrended Value")
dt_3 <- autoplot(sales_add_detrend) + ggtitle("Additive Detrend") + xlab("Time") + ylab("Detrended Value")
dt_4 <- autoplot(sales_mul_detrend) + ggtitle("Multiplicative Detrend") + xlab("Time") + ylab("Detrended Value")

png(paste("./figures/smoothed_plots", seasonal_trend_val, "month", "season.png", sep="_"), width = 712, height = 400)
grid.arrange(dt_1, dt_2, dt_3, dt_4, ncol = 2)
dev.off()

# ACF & PACF of Additive and Fractionally Smoothed Data
saveacf(sales_add_detrend, paste("acf", "sales_additive_detrend", seasonal_trend_val, "months", sep = "_"))
saveacf(sales_frac_smooth, paste("acf", "sales_fractional_detrend", seasonal_trend_val, "months", sep = "_"))
saveacf(sales_seasonally, paste("acf", "sales_seasonal_detrend", seasonal_trend_val, "months", sep = "_"))


#####################################################################################
#### Fit Time Series Models
#####################################################################################
if(fitModels == TRUE)
{
    if (seasonal_trend_val == 12) {
        # 12th and 1st difference
        sarima(sales_seasonally, 1, 1, 0, 1, 1, 0, 12) # -4.119955, -4.109563, -5.087425
        sarima(sales_seasonally, 1, 1, 1, 1, 1, 1, 12) # -4.805208, -4.793919, -5.740147
        sarima(sales, 1, 1, 0, 1, 1, 0, 12) # -4.862065, -4.852282, -5.830813
        sarima(sales, 1, 1, 1, 1, 1, 1, 12) # -5.049281, -5.038699, -5.986776
        
        # 12 month additive decomposition
        sarima(sales_add_detrend, 1, 1, 1, 2, 1, 4, 1) # -3.674233, -3.660929, -4.549223
        
        sarima(sales, 1, 1, 1, 2, 1, 4, 12) # -5.154983, -5.141679, -6.029973
    }
    
    if (seasonal_trend_val == 6){
        # 6th and 1st difference
        sarima(sales_seasonally, 1, 1, 1, 0, 1, 4, 6) # -4.155934, -4.14377, -5.060303
        sarima(sales_seasonally, 1, 1, 1, 0, 1, 8, 6) # -4.593377, -4.577504, -5.433991
        
        sarima(sales, 1, 1, 1, 0, 1, 4, 6) # -4.389765, -4.378013, -5.296007
        sarima(sales, 1, 1, 1, 0, 1, 8, 6) # -4.642834, -4.627584, -5.486572
    }
}

#####################################################################################
#### Forecast Best Models
#####################################################################################
if(forecastModels == TRUE)
{
    # 12 Month Trend Forecasting
    
    if (seasonal_trend_val == 12) 
    {
        png("./figures/sarima_12_month_predict_1.png", width = 712, height = 400)
        sarima.for(sales, 12, 1, 1, 1, 1, 1, 1, 12)
        dev.off()
        
        png("./figures/sarima_12_month_predict_2.png", width = 712, height = 400)
        sarima.for(sales, 12, 1, 1, 1, 2, 1, 4, 12)
        dev.off()
    }
    
    
    
    # 6 Month Trend Forecasting
    if (seasonal_trend_val == 6) {
        png("./figures/sarima_6_month_predict_1.png", width = 712, height = 400)
        sarima.for(sales, 12, 1, 1, 1, 0, 1, 4, 6)
        dev.off()
        
        png("./figures/sarima_6_month_predict_2.png", width = 712, height = 400)
        sarima.for(sales, 12, 1, 1, 1, 0, 1, 8, 6)
        dev.off()
        
    }
    
}

