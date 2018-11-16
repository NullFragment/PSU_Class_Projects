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
              "base2grob")
ipak(packages)

###############################################################################
#### Problem 1
###############################################################################
########################################
#### Data Input
########################################
widgets=read.table("widgets.txt",header=TRUE)
Batch=as.factor(widgets$Batch)
Supplier=widgets$Supplier
WidgetSize=widgets$WidgetSize

widgetTable=table(widgets[,c("Supplier","Batch")])

########################################
## Part C
########################################
widgetNested = aov(WidgetSize ~ Supplier + Supplier:Batch)
anova(widgetNested)

########################################
## Part E
########################################
widgetsLsmSupplier = lsmeans(widgetNested, ~ Supplier)
cld(widgetsLsmSupplier, alpha=0.05)

widgetsLsmInteract = lsmeans(widgetNested, ~ Supplier:Batch)
cld(widgetsLsmInteract, alpha=0.05)
