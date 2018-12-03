###############################################################################
#### Setup
###############################################################################
## Install and load libraries
# ipak function taken from: https://gist.github.com/stevenworthington/3178163
ipak = function(pkg) {
  new.pkg = pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}
packages = c("ggplot2","ggplotify", "reshape2", "gridExtra", "TSA", "astsa",
              "orcutt", "nlme", "fGarch", "vars", "lsmeans", "multcompView",
              "base2grob", "lme4", "lmerTest")
ipak(packages)

###############################################################################
#### Problem 3
###############################################################################
########################################
#### Data Input
########################################
displays=read.table("displays.dat",header=TRUE)
displays$store=as.factor(displays$store)
displays$week=as.factor(displays$week)
displays

########################################
#### Model
########################################
displayModel = lmer(sales ~ (1|store) + display, data=displays)
anova(displayModel)
ranova(displayModel)

displayLSM = lsmeans(displayModel, ~ display)
contrast(displayLSM, method="pairwise")
