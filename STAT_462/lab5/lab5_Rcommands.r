# Import data in R
nutrient=read.table('nutrient.txt',header=TRUE,sep='\t')
# IMPORTANT: CHECK sep in the txt file!!!

# Select all the rows for which protein >50
nutrient50=nutrient[nutrient$protein>50,]



attach(nutrient)

# linear model for y=iron and x=calcium
lm_fit=lm(iron~calcium)

# obtain the residuals
lm_fit$residuals

# obtain the coefficients
lm_fit$coefficients

# P(T>2) for a Student t with 50 degrees of freedom
pt(2,df=50,lower.tail=FALSE)

# P(F>2) for a Fdistribution with 5 and 50 degrees of freedom
pf(2,df1=5,df2=50,lower.tail=FALSE)

detach(nutrient)

