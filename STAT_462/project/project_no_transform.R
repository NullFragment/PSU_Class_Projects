####################################################################
#### Setup
####################################################################
## Install and load libraries
ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

packages <- c("ggplot2", "reshape2", "gridExtra", "leaps", "car", "MASS", 
              "quantreg")
ipak(packages)

## Load and label data
baseball = read.table("baseball.dat.txt")
names(baseball) <- c("Salary", "BattingAverage", "OBP", "Runs", "Hits", 
                     "Doubles", "Triples", "HomeRuns", "RBI", "Walks", "StrikeOuts", "StolenBases", 
                     "Errors", "FreAgtEli", "FreeAgent", "ArbEli", "Arbitration", "Name")

## Add singles data
baseball$Singles <- baseball$Hits - baseball$Doubles - baseball$Triples - 
  baseball$HomeRuns

## Reorder columns
# baseball <- baseball[,c('Salary','BattingAverage','OBP','Runs','Hits','Singles','Doubles','Triples','HomeRuns','RBI','Walks','StrikeOuts','StolenBases','Errors','FreAgtEli',
#                         'FreeAgent','ArbEli','Arbitration','Name')]

## Remove measurement error and some outliers
baseball = baseball[-c(22, 25, 26, 52, 76, 80, 135, 153, 154, 205, 218, 
                       269, 284, 297, 303, 322), ]
baseball = subset(baseball, baseball$Singles > 0)

## Display data summary without categorical variables
summary(baseball[1:13])

## Compute Correlation Matrix
corr_mat <- cor(baseball[1:17])

####################################################################
#### Initial Data Exploration
####################################################################
## Create Scatter Plots of Data
plot(baseball[1:13])

## Generate Histograms of data
ggplot(data = melt(baseball[1:13]), aes(x = value)) + facet_wrap(~variable, 
                                                                 scales = "free_x") + geom_histogram(bins = 50)

## Free un-necessary memory
# rm(p1, p2, p3, p4, p5)
# rm(subset_salary_expanded, subset_obp)
# rm(j, X_obp, x_tmp, y_tmp, R2)
# invisible(gc())

####################################################################
#### Model Selection
####################################################################
## Fit inital linear models
lm_full = lm(Salary ~ . - Name - Singles, data = baseball)
lm_red = lm(Salary ~ . - Name - OBP - Singles, data = baseball)

## Use LEAPS for Model Selection Exploration
select_full = regsubsets(Salary ~ . - Name - Singles, method = "exhaustive", 
                         nbest = 1, nvmax = 16, data = baseball)
select_red = regsubsets(Salary ~ . - Name - Singles - OBP, method = "exhaustive", 
                        nbest = 1, nvmax = 16, data = baseball)
full_sum = summary(select_full)
red_sum = summary(select_red)

## Display Data
full_sum$which
red_sum$which

## Useful Values
totalSS = sum((baseball$Salary - mean(baseball$Salary))^2)
n = nrow(baseball)
sigma_hat_full = summary(lm_full)$sigma

####################################################################
#### Full Model AIC, BIC, Mallows C, Adj R2
####################################################################
## Necessary Values
p_full = 17
p_f = 2:p_full
RSS_full = full_sum$rss

## Adjusted R2
R2_adj_full = 1 - (RSS_full/(n - p_f))/(totalSS/(n - 1))
plot(p_f, R2_adj_full, xlab = "Number of betas", ylab = "Adjusted R-squared")

## Mallow's C
C_p_full = RSS_full/(sigma_hat_full^2) + 2 * p_f - n
C_p_diff_full = abs(abs(C_p_full) - p_f)
plot(p_f, C_p_full, xlab = "Number of betas", ylab = "Mallow's Cp")
abline(0, 1)

## AIC
aic_full = n * log(RSS_full/n) + 2 * p_f
plot(p_f, aic_full, xlab = "Number of betas", ylab = "AIC")

## BIC
bic_full = n * log(RSS_full/n) + p_f * log(n)
plot(p_f, bic_full, xlab = "Number of betas", ylab = "BIC")

## Suggested Number of Betas
r2_f = which.max(R2_adj_full) + 1
mc_f = which.min(C_p_diff_full[1:15]) + 1
aic_f = which.min(aic_full) + 1
bic_f = which.min(bic_full) + 1
full_selection_betas = rbind(c(r2_f, mc_f, aic_f, bic_f))

####################################################################
#### Reduced Model AIC, BIC, Mallows C, Adj R2
####################################################################
## Necessary Values
p_red = 16
p_r = 2:p_red
RSS_red = red_sum$rss

## Adjusted R2
R2_adj_red = 1 - (RSS_red/(n - p_r))/(totalSS/(n - 1))
plot(p_r, R2_adj_red, xlab = "Number of betas", ylab = "Adjusted R-squared")

## Mallow's C
C_p_red = RSS_red/(sigma_hat_full^2) + 2 * p_r - n
C_p_diff_red = abs(abs(C_p_red) - p_r)
plot(p_r, C_p_red, xlab = "Number of betas", ylab = "Mallow's Cp")
abline(0, 1)

## AIC
aic_red = n * log(RSS_red/n) + 2 * p_r
plot(p_r, aic_red, xlab = "Number of betas", ylab = "AIC")

## BIC
bic_red = n * log(RSS_red/n) + p_r * log(n)
plot(p_r, bic_red, xlab = "Number of betas", ylab = "BIC")


## Suggested Number of Betas
r2_r = which.max(R2_adj_red) + 1
mc_r = which.min(C_p_diff_red[1:14]) + 1
aic_r = which.min(aic_red) + 1
bic_r = which.min(bic_red) + 1
red_selection_betas = rbind(c(r2_r, mc_r, aic_r, bic_r))

####################################################################
#### Model Selection Summary and VIF Analysis
####################################################################
## Number of Betas suggested by each method
Updated <- c(7, 10, 10, 7)  #Determined for both full and reduced 

selection_matrix_auto = matrix(rbind(full_selection_betas, red_selection_betas), 
                               ncol = length(full_selection_betas))
selection_matrix_user = matrix(rbind(Updated, Updated), ncol = length(Updated))
colnames(selection_matrix_auto) <- c("Adj R2", "MallowC", "AIC", "BIC")
rownames(selection_matrix_auto) <- c("Full", "Reduced")
colnames(selection_matrix_user) <- c("Adj R2", "MallowC", "AIC", "BIC")
rownames(selection_matrix_user) <- c("Full", "Reduced")

## Full Model Selection Results
lm_full_aic = lm(Salary ~ Runs + Hits + HomeRuns + RBI + StrikeOuts + StolenBases + 
                   FreAgtEli + FreeAgent + ArbEli, data = baseball)
lm_full_bic = lm(Salary ~ HomeRuns + RBI + StrikeOuts + StolenBases + FreAgtEli + 
                   ArbEli, data = baseball)

## Reduced (No OBP) Model Selection Results
lm_red_aic = lm(Salary ~ Runs + Hits + HomeRuns + RBI + StrikeOuts + StolenBases + 
                  FreAgtEli + FreeAgent + ArbEli, data = baseball)
lm_red_bic = lm(Salary ~ HomeRuns + RBI + StrikeOuts + StolenBases + FreAgtEli + 
                  ArbEli, data = baseball)

## NOTE: lm_full_bic and lm_red_r2 are the same model!

######################
# Full models
######################
## Full AIC VIF
vif(lm_full_aic)
which.max(vif(lm_full_aic))

lm_full_aic = update(lm_full_aic, . ~ . - RBI)
vif(lm_full_aic)
which.max(vif(lm_full_aic))

lm_full_aic = update(lm_full_aic, . ~ . - Runs)
vif(lm_full_aic)

## Full BIC VIF
vif(lm_full_bic)
which.max(vif(lm_full_bic))

lm_full_bic = update(lm_full_bic, . ~ . - RBI)
vif(lm_full_bic)
which.max(vif(lm_full_bic))

####################################################################
#### Adjusted Models after VIF and final model selection
####################################################################
## After selection lm_red_r2 and lm_full_bic are the same
summary(lm_full_aic)  # Adj R2 Value = 0.6732
summary(lm_full_bic)  # Adj R2 Value = 0.6459

adj_r2_sum = c(summary(lm_full_aic)$adj.r.squared, summary(lm_full_bic)$adj.r.squared)
names(adj_r2_sum) <- c("lm_full_aic", "lm_full_bic")

lm_best = lm_full_aic
## lm_full_aic is the best model. OBP was not selected for any models by exhaustive model selection.
## Model: Salary ~ Hits + Triples + HomeRuns + StrikeOuts + StolenBases + Errors + FreeAgentEligibility + FreeAgent + ArbitrationEligibility + Arbitration
## Categorical predictors have the highest values for the model. Arbitration eligibility has an extremely high impact, which makes sense
## because salaries are determined through arbitration.

####################################################################
#### Final Model Plots/Tests
####################################################################
## Shapiro-Wilk
shapiro_best = shapiro.test(lm_best$residuals)

## R-plots
plot(lm_best)
plot(lm_best, which = 1)
plot(lm_best, which = 2)
plot(lm_best, which = 3)
plot(lm_best, which = 4)
plot(lm_best, which = 5)
shapiro.test(lm_best$residuals)

means = data.frame(rbind(colMeans(baseball[1:17])))
predict(lm_best, means, interval = "predict")
