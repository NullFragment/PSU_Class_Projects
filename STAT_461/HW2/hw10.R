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

packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa", "orcutt", 
              "nlme", "fGarch", "vars")
ipak(packages)

####################################################################
#### Problem 1
####################################################################
## Load Data
hw_data = read.csv(file="HW10sim.csv", header=TRUE, sep=",")
hw_ts = ts(hw_data)

## Part A
png("./figures/p1_a_1.png", width = 1024, height = 576)
plot(hw_ts)
dev.off()

png("./figures/p1_a_2.png", width = 1024, height = 576)
acf2(hw_ts)
dev.off()

# Both the time series plot, and ACF/PACF plots show nothing but what appears to
# be random noise.

## Part B
hw_ts_sq = hw_ts^2
png("./figures/p1_b_1.png", width = 1024, height = 576)
plot(hw_ts_sq)
dev.off()

png("./figures/p1_b_2.png", width = 1024, height = 576)
acf2(hw_ts_sq)
dev.off()

# Possibly suggests an AR(1) or ARMA(1,1)?

## Part C
hw_ts_centered = hw_ts - mean(hw_ts)
arch1 = garchFit(~garch(1,0), hw_ts_centered)
summary(arch1)
garch11 = garchFit(~garch(1,1), hw_ts_centered, include.mean=F)
summary(garch11)

# GARCH has lower AIC, BIC, SIC, HQIC
# Normality tests show that residuls have a normal distribution

## Part D
# y = \sigma_{t}\epsilon_{t}
# \sigma_{t} = sqrt{0.09622 + 0.31894y^2_{t-1} + 0.60802\sigma^2_{t-1}}
# \epsilon \overset{iid}{\sim} N(0,1)

## Part E
predict(garch11)

####################################################################
#### Problem 2
####################################################################
## Load Data
flourdata = read.csv(file="flour.dat", header=TRUE, sep="\t")
flourdata = flourdata[,2:4]

## Part A
flour_var1_none = VAR(flourdata, p=1, type="none") # 0.9973
flour_var1_both = VAR(flourdata, p=1, type="both") # 0.9308
flour_var1_const = VAR(flourdata, p=1, type="const") # 0.9311
flour_var1_trend = VAR(flourdata, p=1, type="trend") # 0.9974
summary(flour_var1_none) # 0.9971, 0.9975, 0.9977
summary(flour_var1_both) # 0.9252, 0.9271, 0.9252
summary(flour_var1_const) #0.9260, 0.9275, 0.9232
summary(flour_var1_trend) #0.9971, 0.9974, 0.9977


# \hat{B}_{t} = 1.05330 B_{t-1} - 0.30720 M_{t-1} + 0.24882 K_{t-1}
# \hat{M}_{t} = 0.14485 B_{t-1} + 0.56676 M_{t-1} + 0.27974 K_{t-1}
# \hat{K}_{t} = 0.2423 B_{t-1} - 0.4384 M_{t-1} + 1.1886 K_{t-1}

## Part B
flour_var2_none = VAR(flourdata, p=2, type="none") # 0.9973
flour_var2_both = VAR(flourdata, p=2, type="both") # 0.9308
flour_var2_const = VAR(flourdata, p=2, type="const") # 0.9311
flour_var2_trend = VAR(flourdata, p=2, type="trend") # 0.9974
summary(flour_var2_none)
summary(flour_var2_both)
summary(flour_var2_const)
summary(flour_var2_trend)

# \hat{K}_{t} = -0.09426 t - 0.19763 B_{t-1} + 0.59703 M_{t-1} + 0.81700 K_{t-1} + 1.13432 B_{t-2} - 1.67339 M_{t-2} + 0.33416 K_{t-2}


## Part C
n = 100
k = 3
p = 1
BIC1 = log(det(summary(flour_var1_none)$covres)) + (k^2*p*log(n)/n) 

p=2
BIC2 = log(det(summary(flour_var2_none)$covres)) + (k^2*p*log(n)/n) 

# The VAR1 with none is better, as the BIC is slightly lower

## Part D
png("./figures/p2_d_1.png", width = 1024, height = 576)
acf(residuals(flour_var1_none)[,1])
dev.off()
png("./figures/p2_d_2.png", width = 1024, height = 576)
acf(residuals(flour_var1_none)[,2])
dev.off()
png("./figures/p2_d_3.png", width = 1024, height = 576)
acf(residuals(flour_var1_none)[,3])
dev.off()

acf(residuals(flour_var1_none))
