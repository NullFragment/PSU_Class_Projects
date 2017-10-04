# Import data in R
nutrient=read.table('nutrient.txt',header=TRUE,sep='\t')
# IMPORTANT: CHECK sep in the txt file!!!


attach(nutrient)

# compute the quantile of order 0.75 for the variable protein
q75=quantile(protein,probs=0.75)
q75

# compute the quantiles of order 0.25, 0.50 and 0.75 for the variable protein
q_protein=quantile(protein,probs=c(0.25,0.50,0.75))
q_protein


# linear model for y=iron, x1=calcium, x2=protein, x3=vitaminA, x4=vitaminC
lm_fit=lm(iron~calcium+protein+vitaminA+vitaminC)
summary(lm_fit)

# obtain the coefficients
lm_fit$coefficients

# obtain the number of coefficient in the model
length(lm_fit$coefficients)

# obtain sigma_hat
summary(lm_fit)$sigma

# create a new data.frame with new values for each variable
x0=data.frame(calcium=500,
              protein=40,
              vitaminA=600,
              vitaminC=30)
x0

# create a new data.frame with new values for each variable, using the quantiles of order 0.25, 0.50 and 0.75 for each variable
x0=data.frame(calcium=quantile(calcium,probs=c(0.25,0.50,0.75)),
              protein=quantile(protein,probs=c(0.25,0.50,0.75)),
              vitaminA=quantile(vitaminA,probs=c(0.25,0.50,0.75)),
              vitaminC=quantile(vitaminC,probs=c(0.25,0.50,0.75)))
x0

# use the fitted linear model and the new data.frame to compute a point estimate for the mean response 
# corresponding to each row in the new data.frame
predict(lm_fit,new=x0)

# use the fitted linear model and the new data.frame to compute a 99% Confidence Interval for the mean response 
# corresponding to each row in the new data.frame
predict(lm_fit,new=x0,interval='confidence',level=0.99)

# use the fitted linear model and the new data.frame to compute a 99% Prediction Interval for a new response 
# corresponding to each row in the new data.frame
predict(lm_fit,new=x0,interval='prediction',level=0.99)



detach(nutrient)

