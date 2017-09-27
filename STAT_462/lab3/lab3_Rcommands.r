# Import data in R

# Import data in R
nutrient=read.table('nutrient.txt',header=TRUE,sep='\t')
# IMPORTANT: CHECK sep in the txt file!!!

# Look at first 6 observations
head(nutrient)

# Scatterplot matrix (scatterplot between each pair of variables)
plot(nutrient)

# Correlation matrix (correlation between each pair of variables)
cor(nutrient)



# Make the variables in the data.frame directly accessible
attach(nutrient)

# Mean of calcium
c_mean=mean(calcium)
c_mean

# Sum of all observations in the variable calcium
c_sum=sum(calcium)
c_sum

# Square of all observations in the variable calcium
c2=calcium^2
c2

# Create a matrix with 1 as first column and the variable calcium as second column
M=cbind(1,calcium)
M

# Transpose of M
t(M)

# Solve the equation system A*b=c
# b=solve(A,c)

# Inverse matrix of A
# A_inv=solve(A)


# Scatterplot of vitaminA vs calcium
plot(iron,calcium)

# Add the line 300+25*calcium to the plot
abline(300,25)
# Change color at the line
abline(300,25,col='red')

# sample size for calcium
length(calcium)

# 85th percentile of calcium
quantile(calcium,probs=0.85)

# Detach the data.frame
detach(nutrient)

