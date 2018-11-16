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
#### Data Input P1
########################################
fert=c(rep("control", 12), rep("f1", 12),
        rep("f2", 12), rep("f3", 12))
species=c(rep(c(rep("SppA", 6), rep("SppB", 6)),4))
height=c(21.0, 19.5, 22.5, 21.5, 20.5, 21.0,
          23.7, 23.8, 23.8, 23.7, 22.8, 24.4,
          32.0, 30.5, 25.0, 27.5, 28.0, 28.6,
          30.1, 28.9, 30.9, 34.4, 32.7, 32.7,
          22.5, 26.0, 28.0, 27.0, 26.5, 25.2,
          30.6, 31.1, 28.1, 34.9, 30.1, 25.5,
          28.0, 27.5, 31.0, 29.5, 30.0, 29.2,
          36.1, 36.6, 38.7, 37.1, 36.8, 37.1)
fertData=data.frame(fert,species,height)

########################################
## Part B
########################################
fertModelNorm = aov(height ~ fert+species+fert:species, data=fertData)
fertModelInvt = aov(1/(height) ~ fert+species+fert:species, data=fertData)
fertModelSqrt = aov(sqrt(height) ~ fert+species+fert:species, data=fertData)
fertModelLog = aov(log(height) ~ fert+species+fert:species, data=fertData)

####################
# Res vs Fit Plots
####################
png("./figures/fert/fertNorm-ResidualVsFitted.png", width = 512, height = 288)
  plot(fertModelNorm,1)
dev.off()

png("./figures/fert/fertInvt-ResidualVsFitted.png", width = 512, height = 288)
  plot(fertModelInvt,1)
dev.off()

png("./figures/fert/fertSqrt-ResidualVsFitted.png", width = 512, height = 288)
  plot(fertModelSqrt,1)
dev.off()

png("./figures/fert/fertLog-ResidualVsFitted.png", width = 512, height = 288)
  plot(fertModelLog,1)
dev.off()

resPlots = lapply(sprintf("./figures/fert/fert%s-ResidualVsFitted.png", c("Norm","Invt","Sqrt","Log")), png::readPNG)
resGrid = lapply(resPlots, grid::rasterGrob)

png("./figures/fertAll-ResidualVsFitted.png", width = 1024, height = 576)
  gridExtra::grid.arrange(grobs=resGrid)
dev.off()


####################
# Res vs Fit Plots
####################
png("./figures/fert/fertNorm-QQ.png", width = 512, height = 288)
  plot(fertModelNorm,2)
dev.off()

png("./figures/fert/fertInvt-QQ.png", width = 512, height = 288)
  plot(fertModelInvt,2)
dev.off()

png("./figures/fert/fertSqrt-QQ.png", width = 512, height = 288)
  plot(fertModelSqrt,2)
dev.off()

png("./figures/fert/fertLog-QQ.png", width = 512, height = 288)
  plot(fertModelLog,2)
dev.off()

qqPlots = lapply(sprintf("./figures/fert/fert%s-QQ.png", c("Norm","Invt","Sqrt","Log")), png::readPNG)
qqGrid = lapply(qqPlots, grid::rasterGrob)

png("./figures/fertAll-QQ.png", width = 1024, height = 576)
  gridExtra::grid.arrange(grobs=qqGrid)
dev.off()

########################################
## Part C
########################################
summary(fertModelLog) # Interactions insignificant -> Pairwise

png("./figures/fertInteraction.png", width = 1024, height = 576)
  interaction.plot(x.factor = fertData$fert, trace.factor = fertData$species,
                 response = fertData$height, type ="b",col = 2:3,
                 xlab ="fert", ylab ="height", trace.label ="species")
dev.off()
  
fertModelLogLM=lsmeans(fertModelLog, ~ fert )
contrast(fertModelLogLM, method="pairwise")


###############################################################################
#### Problem 2
###############################################################################
########################################
#### Data Input P2
########################################
A=c(rep(1, 10), rep(2, 10))
B=rep(c(c(rep(1, 5), rep(2, 5))), 2)
resp=c(12.9, 11.3, 11.7, 12.1, 12.3,
        13.7, 12.8, 13.6, 13.1, 13.5,
        14.2, 14.5, 13.9, 13.6, 14.4,
        13.5, 13.1, 13.3, 13.1, 13.4)
respData=data.frame(A, B, resp)

########################################
## Part B
########################################
respModelNorm = aov(resp ~ A+B+A:B, data=respData)
respModelInvt = aov(1/(resp) ~ A+B+A:B, data=respData)
respModelSqrt = aov(sqrt(resp) ~ A+B+A:B, data=respData)
respModelLog = aov(log(resp) ~ A+B+A:B, data=respData)

####################
# Res vs Fit Plots
####################
png("./figures/resp/respNorm-ResidualVsFitted.png", width = 512, height = 288)
  plot(respModelNorm,1)
dev.off()

png("./figures/resp/respInvt-ResidualVsFitted.png", width = 512, height = 288)
  plot(respModelInvt,1)
dev.off()

png("./figures/resp/respSqrt-ResidualVsFitted.png", width = 512, height = 288)
  plot(respModelSqrt,1)
dev.off()

png("./figures/resp/respLog-ResidualVsFitted.png", width = 512, height = 288)
  plot(respModelLog,1)
dev.off()

resPlots = lapply(sprintf("./figures/resp/resp%s-ResidualVsFitted.png", c("Norm","Invt","Sqrt","Log")), png::readPNG)
resGrid = lapply(resPlots, grid::rasterGrob)

png("./figures/respAll-ResidualVsFitted.png", width = 1024, height = 576)
  gridExtra::grid.arrange(grobs=resGrid)
dev.off()


####################
# Res vs Fit Plots
####################
png("./figures/resp/respNorm-QQ.png", width = 512, height = 288)
  plot(respModelNorm,2)
dev.off()

png("./figures/resp/respInvt-QQ.png", width = 512, height = 288)
  plot(respModelInvt,2)
dev.off()

png("./figures/resp/respSqrt-QQ.png", width = 512, height = 288)
  plot(respModelSqrt,2)
dev.off()

png("./figures/resp/respLog-QQ.png", width = 512, height = 288)
  plot(respModelLog,2)
dev.off()

qqPlots = lapply(sprintf("./figures/resp/resp%s-QQ.png", c("Norm","Invt","Sqrt","Log")), png::readPNG)
qqGrid = lapply(qqPlots, grid::rasterGrob)

png("./figures/respAll-QQ.png", width = 1024, height = 576)
  gridExtra::grid.arrange(grobs=qqGrid)
dev.off()

########################################
## Part C
########################################
summary(respModelNorm) # Interactions significant -> Pairwise impossible

png("./figures/respInteraction.png", width = 1024, height = 576)
  interaction.plot(x.factor = respData$A, trace.factor = respData$B,
                 response = respData$resp, type ="b",col = 2:3,
                 xlab ="A", ylab ="resp", trace.label ="B")
dev.off()