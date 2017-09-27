# Import data in R
nutrient=read.table('nutrient.txt',header=TRUE,sep='\t')
# IMPORTANT: CHECK sep in the txt file!!!

# Look at first 6 observations
head(nutrient)

# Make the variables in the data.frame directly accessible
attach(nutrient)

# Compute summary statistics, mean, sd, sample size...
summary(calcium)
mean(calcium)
sd(calcium)
length(calcium)

# Scatterplot and correlation
plot(iron,protein)
cor(iron,protein)

# Detach the data.frame
detach(nutrient)




# Quantile function for normal distribution (arguments: probability, mean, sd)
?qnorm
qnorm(0.975) 
# Quantile function for t distribution (arguments: probability, degrees of freedom)
?qt
qt(0.975,100)
# Distribution function for normal distribution (arguments: quantile, mean, sd)
pnorm(1.96) 
# Distribution function for t distribution (arguments: quantile, degrees of freedom)
pt(2,100)

