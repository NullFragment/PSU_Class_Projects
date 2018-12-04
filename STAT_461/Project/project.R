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
#### Randomly assign 3 of each pair of treatments to spots on the tray
###############################################################################
options = c(rep("MR",3),rep("MN",3),rep("CR",3),rep("CN",3))
tray1 = sample(options)
tray2 = sample(options)
tray1
tray2

###############################################################################
## Read in Data
###############################################################################
cookies=read.table("cookies.txt",header=TRUE)

###############################################################################
## Plot Data in Boxplots
###############################################################################
plotWidth=512;
plotHeight=512;

######################
## Individual Plots
######################

png("./figures/boxplots/boxplot-1-butter.png", width = plotWidth, height = plotHeight)
  boxplot(size ~ butter, data=cookies, main = "Butter")
dev.off()

png("./figures/boxplots/boxplot-1-chilled.png", width = plotWidth, height = plotHeight)
  boxplot(size ~ chilled, data=cookies, main = "Dough Temperature")
dev.off()
  
png("./figures/boxplots/boxplot-1-tray.png", width = plotWidth, height = plotHeight)
  boxplot(size ~ tray, data=cookies, main = "Tray Position")
dev.off()

######################
## Pair Plots
######################
png("./figures/boxplots/boxplot-2-butter-chilled.png", width = plotWidth, height = plotHeight)
  boxplot(size ~ chilled + butter, data=cookies, main = "Butter & Temperature")
dev.off()
  
png("./figures/boxplots/boxplot-2-butter-tray.png", width = plotWidth, height = plotHeight)
  boxplot(size ~ tray + butter, data=cookies, main = "Butter & Tray")
dev.off()
  
png("./figures/boxplots/boxplot-2-chilled-tray.png", width = plotWidth, height = plotHeight)
  boxplot(size ~ tray + chilled, data=cookies, main = "Temperature & Tray")
dev.off()

######################
## Full Plot
######################
png("./figures/boxplot-full.png", width = 2*plotWidth, height = 2*plotHeight)
  boxplot(size ~ tray + chilled + butter, data=cookies)
dev.off()

######################
## Compile Plots
######################
singleBoxplots = lapply(sprintf("./figures/boxplots/boxplot-1-%s.png", c("butter","chilled","tray")), png::readPNG)
singleBoxGrid = lapply(singleBoxplots, grid::rasterGrob)

doubleBoxPlots = lapply(sprintf("./figures/boxplots/boxplot-2-%s.png", c("butter-chilled","butter-tray","chilled-tray")), png::readPNG)
doubleBoxGrid = lapply(doubleBoxPlots, grid::rasterGrob)

png("./figures/boxplot-individuals.png", width = 3*plotWidth, height = plotHeight)
  gridExtra::grid.arrange(ncol=3, grobs=singleBoxGrid)
dev.off()

png("./figures/boxplot-pairs.png", width = 3*plotWidth, height = plotHeight)
  gridExtra::grid.arrange(ncol=3, grobs=doubleBoxGrid)
dev.off()

###############################################################################
## Create Model
###############################################################################
cookieFullModel = aov(size ~ tray + chilled + butter + tray:chilled + tray:butter + chilled:butter, data=cookies)
anova(cookieFullModel)

cookieMainModel = aov(size ~ tray + chilled + butter, data=cookies)
anova(cookieMainModel)

#######################
# Check Normality 
#######################
png("./figures/normality-residuals.png", width = 3*plotWidth, height = plotHeight)
  plot(cookieFullModel,1)
dev.off()

png("./figures/normality-qq.png", width = 3*plotWidth, height = plotHeight)
  plot(cookieFullModel,2)
dev.off()

#######################
# Examine Interactions 
#######################
png("./figures/interaction-tray-butter.png", width = 3*plotWidth, height = plotHeight)
  interaction.plot(x.factor = cookies$tray, trace.factor = cookies$butter,
                   response = cookies$size, type ="b",col = 2:3,
                   xlab ="Tray", ylab ="Size", trace.label ="Butter")
dev.off()

png("./figures/interaction-chilled-butter.png", width = 3*plotWidth, height = plotHeight)
  interaction.plot(x.factor = cookies$chilled, trace.factor = cookies$butter,
                   response = cookies$size, type ="b",col = 2:3,
                   xlab ="Chilled", ylab ="Size", trace.label ="Butter")
dev.off()
  