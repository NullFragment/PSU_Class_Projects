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
#### Problem 1
####################################################################
meds = c(rep('A',2),rep('B',2),rep('Control',2))
delta = c(-14,-4,5,-1,-2,6)
bpData = data.frame(delta, meds)

# Calculate ANOVA table and Pairwise Comparison
bpModel = aov(delta ~ meds, data=bpData)
anova(bpModel)
bpLSM = lsmeans(bpModel, ~ meds)
contrast(bpLSM, method='pairwise')

####################################################################
#### Problem 2
####################################################################
# From HW1
loss = c(-.3,-.1,-.14,.4,2.63,2.61,2.41,3.15,1.86,2.03,2.26,1.82)
type = c(rep('reg',4), rep('deo',4), rep('moi',4))
losses = data.frame(loss, type)

# Calculate ANOVA table and Pairwise Comparison
lossModel = aov(loss~type, data=losses)
anova(lossModel)
lossLSM = lsmeans(lossModel, ~type)
contrast(lossLSM, method='pairwise')


####################################################################
#### Problem 3
####################################################################
# From HW4
time = c(38.14, 38.20, 38.31, 38.14, 38.29, 38.17, 38.20,
         38.28, 38.17, 38.08, 38.25, 38.18, 38.03, 37.95, 38.26, 38.30, 38.21,
         38.17, 38.13, 38.16, 38.30, 38.34, 38.34, 38.17, 38.18, 38.09, 38.06,
         38.14, 38.30, 38.21, 38.04, 38.37)
presses = c(rep('0',7), rep('1',10), rep('2',10), rep('3',5))
lights = data.frame(time, presses)

# Calculate ANOVA table and Pairwise Comparison
lightModel = aov(time~presses, data=lights)
anova(lightModel)
lightLSM = lsmeans(lightModel, ~presses)
contrast(lightLSM, method='pairwise')
