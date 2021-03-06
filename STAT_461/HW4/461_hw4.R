####################################################################
#### Setup
####################################################################
## Install and load libraries
# ipak function taken from: https://gist.github.com/stevenworthington/3178163
ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}
packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa", "orcutt",
              "nlme", "fGarch", "vars", "lsmeans")
ipak(packages)

####################################################################
#### Problem 2
####################################################################
# From HW1
loss = c(-.3,-.1,-.14,.4,2.63,2.61,2.41,3.15,1.86,2.03,2.26,1.82)
type = c(rep('reg',4), rep('deo',4), rep('moi',4))
losses = data.frame(loss, type)

# Part D
aov_losses = aov(loss~type)
lsmeans_losses = lsmeans(aov_losses, "type")
lsmeans_losses

# Part B & C Re-calculation:
contrast(lsmeans_losses, list("deodorant only"=c(0,1,0), "t1 - 1/2(t2+t3)"=c(1,-0.5,-0.5)))

####################################################################
#### Problem 3
####################################################################
time = c(38.14, 38.20, 38.31, 38.14, 38.29, 38.17, 38.20,
         38.28, 38.17, 38.08, 38.25, 38.18, 38.03, 37.95, 38.26, 38.30, 38.21,
         38.17, 38.13, 38.16, 38.30, 38.34, 38.34, 38.17, 38.18, 38.09, 38.06,
         38.14, 38.30, 38.21, 38.04, 38.37)
presses = c(rep('0',7), rep('1',10), rep('2',10), rep('3',5))
lights = data.frame(time, presses)

# Part A
png("./figures/p3.png", width = 1024, height = 576)
  boxplot(time ~ presses, main="Boxplot of Pedestrian Light Experiment")
dev.off()

# Part C
aov_lights = aov(time ~ presses)
lsmeans_lights = lsmeans(aov_lights, "presses")
lsmeans_lights

# Part D & E
contrast(lsmeans_lights, list("Part D"=c(-1, 1, 0, 0), "Part E"=c(-1, 1/3, 1/3, 1/3)))



