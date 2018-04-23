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

packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa", "orcutt", "nlme")
ipak(packages)

# png("./figures/interaction.png", width = 1024, height = 576)
# Plot Statment
# dev.off()

####################################################################
#### DATA ANALYSIS
####################################################################
## Load Data
sales_data = read.csv(file="SalesData.csv", header=TRUE, sep=",")
sales = ts(sales_data$Sales)
plot(sales) # Linear trend and seasonality. Detrend with linear transformation.
acf2(sales) # 1st or 2nd order AR appearance with no transformation. 12 time step spike -> 12 month seasonality

sales_diff = diff(sales)
plot(sales_diff)
sales_yearly = diff(sales_diff, 12)
plot(sales_yearly) # Still noisy. Sinusoidal


sales_trend = filter(sales, filter=c(1/12, rep(1/12, 11), 1/24), sides=2)
plot(sales_trend)
sales_detrend = sales - sales_trend
plot(sales_detrend)
acf2(sales_detrend)


sales_arima = arima(sales, order = c(0,1,1))
sales_exp = sales-sales_arima$residuals
sales_detrend_exp = sales - sales_exp
plot(sales_detrend_exp)

sales_freq = ts(sales_data$Sales, freq=12)
sales_decomp_add = decompose(sales_freq, type="additive")
sales_decomp_exp = decompose(sales_freq, type="multiplicative")

plot(sales_decomp_add)
sales_decomp_add_season = sales_decomp_add$seasonal
plot(sales_decomp_add_season)
acf2(sales_decomp_add_season)

test = sarima(sales_decomp_add_season,0,0,0,1,3,1,1)
