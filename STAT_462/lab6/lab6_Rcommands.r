# Import data in R
nutrient=read.table('nutrient.txt',header=TRUE,sep='\t')
# IMPORTANT: CHECK sep in the txt file!!!


attach(nutrient)

# linear model for y=iron, x1=calcium, x2=protein, x3=vitaminA, x4=vitaminC
lm_fit=lm(iron~calcium+protein+vitaminA+vitaminC)
summary(lm_fit)

# obtain the residuals
lm_fit$residuals

# obtain the coefficients
lm_fit$coefficients

# obtain the number of coefficient in the model
length(lm_fit$coefficients)

# P(F>2) for a Fdistribution with 5 and 50 degrees of freedom
pf(2,df1=5,df2=50,lower.tail=FALSE)

detach(nutrient)

