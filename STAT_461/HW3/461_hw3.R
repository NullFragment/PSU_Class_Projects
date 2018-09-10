####################################################################
#### Setup
####################################################################
## Install and load libraries
# ipak function taken from: https://gist.github.com/stevenworthington/3178163
# ipak <- function(pkg) {
#   new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
#   if (length(new.pkg)) 
#     install.packages(new.pkg, dependencies = TRUE)
#   sapply(pkg, require, character.only = TRUE)
# }
# 
# packages <- c("ggplot2", "reshape2", "gridExtra", "TSA", "astsa", "orcutt", 
#               "nlme", "fGarch", "vars")
# ipak(packages)



####################################################################
#### Problem 1
####################################################################
# Set up treatment distribution and experimental units
n=30
treatments = c(rep("1", 10), rep("2", 10), rep("3", 10))
exp_units = 1:n

# Randomize treatments
p1_randomized = sample(treatments)

# Set treatment parameters
mu_1 = 4.7
var_1 = 4
tao_1 = -3
tao_2 = 5
tao_3 = -2

# Generate treatment means
means_1 = rep(NA, n)
means_1[p1_randomized=="1"] = mu_1 + tao_1
means_1[p1_randomized=="2"] = mu_1 + tao_2
means_1[p1_randomized=="3"] = mu_1 + tao_3

# Simulate ANOVA model
p1_sim = means_1 + rnorm(n, mean = 0, sd = sqrt(var_1))

# Plot Data
p1_data = data.frame(exp_units, p1_randomized, p1_sim)

png("./figures/p1.png", width = 1024, height = 576)
  boxplot(p1_sim ~ p1_randomized, main="Boxplot of Simulated ANOVA Data")
dev.off()


####################################################################
#### Problem 2
####################################################################
p2_randomized = sample(treatments)

# Generate treatment means
means_2 = rep(mu_1, n)

# Simulate ANOVA model
p2_sim = means_2 + rnorm(n, mean = 0, sd = sqrt(var_1))

# Plot Data
p2_data = data.frame(exp_units, p2_randomized, p2_sim)

png("./figures/p2.png", width = 1024, height = 576)
  boxplot(p2_sim ~ p2_randomized, main="Boxplot of Reduced ANOVA Data")
dev.off()

####################################################################
#### Problem 3
####################################################################
p3_randomized = sample(treatments)

# Set treatment parameters
var_3 = 50

# Simulate ANOVA model
p3_sim = means_1 + rnorm(n, mean = 0, sd = sqrt(var_3))

# Plot Data
p3_data = data.frame(exp_units, p3_randomized, p3_sim)

png("./figures/p3.png", width = 1024, height = 576)
  boxplot(p3_sim ~ p3_randomized, main="Boxplot of Simulated ANOVA Data")
dev.off()
