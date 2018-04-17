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

packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa", "orcutt", "nlme")
ipak(packages)

####################################################################
#### Problem 1
####################################################################
## Load Data
trt_data = read.csv(file="HW9.csv", header=TRUE, sep=",")
attach(trt_data)

## Part A
png("./figures/interaction.png", width = 1024, height = 576)
interaction.plot(time, factor(trt), Y, lty=c(1:3),lwd=2,ylab="mean of Y", xlab="time", trace.label="Treatment")
dev.off()

# The mean increases for all treatments over time.
# Treatment C seems to be independent of A & B, howerver A & B appear to have an interaction at aroud 1.4 and 2.6

## Part B

treatment_fit_compsym <- gls(Y ~ factor(trt)*factor(time), data=trt_data, corr=corCompSymm(, form= ~ 1 | subject))
treatment_fit_nostruct <- gls(Y ~ factor(trt)*factor(time), data=trt_data, corr=corSymm(, form= ~ 1 | subject), weights = varIdent(form = ~ 1 | time))
treatment_fit_ar1 <- gls(Y ~ factor(trt)*factor(time), data=trt_data, corr=corAR1(, form= ~ 1 | subject))
treatment_fit_ar1het <- gls(Y ~ factor(trt)*factor(time), data=trt_data, corr=corAR1(, form= ~ 1 | subject), weights=varIdent(form = ~ 1 | time))
anova(treatment_fit_compsym, treatment_fit_nostruct, treatment_fit_ar1, treatment_fit_ar1het)


# Best model is compsym

## Part C
anova(treatment_fit_compsym)
