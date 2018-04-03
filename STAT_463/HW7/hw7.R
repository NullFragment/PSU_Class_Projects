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


####################################################################
#### Problem 1
####################################################################

## Part A
# The peak frequency is between 0.08 and 0.09
# The frequency corresponds to about 12 months

## Part B
# The second highest peak range is between 0.01 and 0.05

####################################################################
#### Problem 2
####################################################################
## Read Data
speech_data = read.table("speech.txt", header=T)
speech_ts = ts(speech_data[,2])
png("./figures/speech_ts.png", width = 1024, height = 576)
plot(speech_ts)
dev.off()

speech_regression = lm(Pitch~Time, data = speech_data)
speech_residuals = residuals(speech_regression)
periodogram(speech_residuals)


png("./figures/periodogram.png", width = 1024, height = 576)
periodogram(speech_residuals)
dev.off()


# Primary frequencies are in the .07, .06, .05 range but peaks are approximately every 0.01