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

#packages <- c('ggplot2', 'reshape2', 'gridExtra', 'leaps', 'car', 'rstudioapi')
packages <- c("ggplot2", "reshape2", "gridExtra", "leaps", "car")
ipak(packages)

## Set working dir
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

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

###################################################################################################################################################################################################
## ORIGINAL PART
###################################################################################################################################################################################################

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
plot(p_f, R2_adj_full, xlab = "Number of betas", ylab = "Adjusted R-squared")  # Manual: 9

## Mallow's C
C_p_full = RSS_full/(sigma_hat_full^2) + 2 * p_f - n
C_p_diff_full = abs(abs(C_p_full) - p_f)
plot(p_f, C_p_full, xlab = "Number of betas", ylab = "Mallow's Cp")  # Manual: 8
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
plot(p_r, R2_adj_red, xlab = "Number of betas", ylab = "Adjusted R-squared")  # Manual: 8

## Mallow's C
C_p_red = RSS_red/(sigma_hat_full^2) + 2 * p_r - n
C_p_diff_red = abs(abs(C_p_red) - p_r)
plot(p_r, C_p_red, xlab = "Number of betas", ylab = "Mallow's Cp")  # 15
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
manual_full <- c(9, 8, 9, 8)  #Determined for both full and reduced 
manual_red <- c(8, 15, 9, 8)  #Determined for both full and reduced 

selection_matrix_auto = matrix(rbind(full_selection_betas, red_selection_betas), 
                               ncol = length(full_selection_betas))
selection_matrix_user = matrix(rbind(manual_full, manual_red), ncol = length(manual_red))
colnames(selection_matrix_auto) <- c("Adj R2", "MallowC", "AIC", "BIC")
rownames(selection_matrix_auto) <- c("Full", "Reduced")
colnames(selection_matrix_user) <- c("Adj R2", "MallowC", "AIC", "BIC")
rownames(selection_matrix_user) <- c("Full", "Reduced")

## Full Model Selection Results

## NOTE: AIC and BIC selections for Reduced and Full are the SAME
lm_full_aic = lm(Salary ~ Runs + HomeRuns + RBI + StrikeOuts + StolenBases + 
                   FreAgtEli + FreeAgent + ArbEli, data = baseball)
lm_full_bic = lm(Salary ~ HomeRuns + RBI + StrikeOuts + StolenBases + FreAgtEli + 
                   FreeAgent + ArbEli, data = baseball)
lm_red_mallow = lm(Salary ~ Runs + Hits + Doubles + Triples + HomeRuns + 
                     RBI + Walks + StrikeOuts + StolenBases + Errors + FreAgtEli + FreeAgent + 
                     ArbEli + Arbitration, data = baseball)

######################
# Same models
######################
## AIC VIF
vif(lm_full_aic)
which.max(vif(lm_full_aic))

lm_full_aic = update(lm_full_aic, . ~ . - RBI)
vif(lm_full_aic)
which.max(vif(lm_full_aic))

## BIC VIF
vif(lm_full_bic)
which.max(vif(lm_full_bic))

lm_full_bic = update(lm_full_bic, . ~ . - HomeRuns)
vif(lm_full_bic)
which.max(vif(lm_full_bic))

######################
# Reduced models
######################
vif(lm_red_mallow)
which.max(vif(lm_red_mallow))

lm_red_mallow = update(lm_red_mallow, . ~ . - Runs)
vif(lm_red_mallow)
which.max(vif(lm_red_mallow))

lm_red_mallow = update(lm_red_mallow, . ~ . - RBI)
vif(lm_red_mallow)
which.max(vif(lm_red_mallow))

lm_red_mallow = update(lm_red_mallow, . ~ . - Hits)
vif(lm_red_mallow)
which.max(vif(lm_red_mallow))

####################################################################
#### Adjusted Models after VIF and final model selection
####################################################################
## After selection lm_red_r2 and lm_full_bic are the same
summary(lm_full_aic)  # Adj R2 Value = 0.7301
summary(lm_full_bic)  # Adj R2 Value = 0.7279
summary(lm_red_mallow)  # Adj R2 Value = 0.7229

adj_r2_sum = c(summary(lm_full_aic)$adj.r.squared, summary(lm_full_bic)$adj.r.squared, 
               summary(lm_red_mallow)$adj.r.squared)
names(adj_r2_sum) <- c("lm_full_aic", "lm_full_bic", "lm_red_mallow")

lm_untransformed = lm_full_aic
## lm_full_aic is the best model. OBP was not selected for any models by exhaustive model selection.
## Model: Salary ~ Hits + Triples + HomeRuns + StrikeOuts + StolenBases + Errors + FreeAgentEligibility + FreeAgent + ArbitrationEligibility + Arbitration
## Categorical predictors have the highest values for the model. Arbitration eligibility has an extremely high impact, which makes sense
## because salaries are determined through arbitration.

####################################################################
#### Final Model Plots/Tests
####################################################################
## R-plots
plot(lm_untransformed, which = 1)
plot(lm_untransformed, which = 2)
plot(lm_untransformed, which = 3)
plot(lm_untransformed, which = 4)
plot(lm_untransformed, which = 5)
shapiro.test(lm_untransformed$residuals)

means = data.frame(rbind(colMeans(baseball[1:17])))
means[14:17] <- round(means[14:17])
unt_pred = predict(lm_untransformed, means, interval = "predict")

###################################################################################################################################################################################################
## LOG TRANSFORMED PART
###################################################################################################################################################################################################

####################################################################
#### Model Selection
####################################################################
## Fit inital linear models
lm_log_full = lm(log(Salary) ~ . - Name - Singles, data = baseball)
lm_log_red = lm(log(Salary) ~ . - Name - OBP - Singles, data = baseball)

## Use LEAPS for Model Selection Exploration
select_log_full = regsubsets(log(Salary) ~ . - Name - Singles, method = "exhaustive", 
                             nbest = 1, nvmax = 16, data = baseball)
select_log_red = regsubsets(log(Salary) ~ . - Name - Singles - OBP, method = "exhaustive", 
                            nbest = 1, nvmax = 16, data = baseball)
full_log_sum = summary(select_log_full)
red_log_sum = summary(select_log_red)

## Display Data
full_log_sum$which
red_log_sum$which

## Useful Values
totalSS_log = sum((log(baseball$Salary) - mean(baseball$Salary))^2)
n = nrow(baseball)
sigma_hat_log_full = summary(lm_log_full)$sigma

####################################################################
#### Full Model AIC, BIC, Mallows C, Adj R2
####################################################################
## Necessary Values
p_log_full = 17
p_log_f = 2:p_log_full
RSS_log_full = full_log_sum$rss

## Adjusted R2
R2_adj_log_full = 1 - (RSS_log_full/(n - p_log_f))/(totalSS_log/(n - 1))
plot(p_log_f, R2_adj_log_full, xlab = "Number of betas", ylab = "Adjusted R-squared")

## Mallow's C
C_p_log_full = RSS_log_full/(sigma_hat_log_full^2) + 2 * p_log_f - n
C_p_diff_log_full = abs(abs(C_p_log_full) - p_log_f)
plot(p_log_f, C_p_log_full, xlab = "Number of betas", ylab = "Mallow's Cp")
abline(0, 1)

## AIC
aic_log_full = n * log(RSS_log_full/n) + 2 * p_log_f
plot(p_log_f, aic_log_full, xlab = "Number of betas", ylab = "AIC")

## BIC
bic_log_full = n * log(RSS_log_full/n) + p_log_f * log(n)
plot(p_log_f, bic_log_full, xlab = "Number of betas", ylab = "BIC")

## Suggested Number of Betas
r2_log_f = which.max(R2_adj_log_full) + 1
mc_log_f = which.min(C_p_diff_log_full[1:15]) + 1
aic_log_f = which.min(aic_log_full) + 1
bic_log_f = which.min(bic_log_full) + 1
log_full_selection_betas = rbind(c(r2_log_f, mc_log_f, aic_log_f, bic_log_f))

####################################################################
#### Reduced Model AIC, BIC, Mallows C, Adj R2
####################################################################
## Necessary Values
p_log_red = 16
p_log_r = 2:p_log_red
RSS_log_red = red_log_sum$rss

## Adjusted R2
R2_adj_red = 1 - (RSS_log_red/(n - p_log_r))/(totalSS_log/(n - 1))
plot(p_log_r, R2_adj_red, xlab = "Number of betas", ylab = "Adjusted R-squared")

## Mallow's C
C_p_log_red = RSS_log_red/(sigma_hat_log_full^2) + 2 * p_log_r - n
C_p_diff_log_red = abs(abs(C_p_log_red) - p_log_r)
plot(p_log_r, C_p_log_red, xlab = "Number of betas", ylab = "Mallow's Cp")
abline(0, 1)

## AIC
aic_log_red = n * log(RSS_log_red/n) + 2 * p_log_r
plot(p_log_r, aic_log_red, xlab = "Number of betas", ylab = "AIC")

## BIC
bic_log_red = n * log(RSS_log_red/n) + p_log_r * log(n)
plot(p_log_r, bic_log_red, xlab = "Number of betas", ylab = "BIC")

## Suggested Number of Betas
r2_log_r = which.max(R2_adj_red) + 1
mc_log_r = which.min(C_p_diff_log_red[1:14]) + 1
aic_log_r = which.min(aic_log_red) + 1
bic_log_r = which.min(bic_log_red) + 1
log_red_selection_betas = rbind(c(r2_log_r, mc_log_r, aic_log_r, bic_log_r))

####################################################################
#### Model Selection Summary and VIF Analysis
####################################################################
## Number of Betas suggested by each method

selection_matrix_log_auto = matrix(rbind(log_full_selection_betas, log_red_selection_betas), 
                                   ncol = length(log_full_selection_betas))
colnames(selection_matrix_log_auto) <- c("Adj R2", "MallowC", "AIC", "BIC")
rownames(selection_matrix_log_auto) <- c("Full", "Reduced")

#   (Intercept) BattingAverage   OBP  Runs  Hits Doubles Triples HomeRuns   RBI Walks StrikeOuts StolenBases Errors FreAgtEli FreeAgent ArbEli Arbitration
#5         TRUE          FALSE FALSE FALSE  TRUE   FALSE   FALSE    FALSE  TRUE FALSE      FALSE       FALSE  FALSE      TRUE      TRUE   TRUE       FALSE
#7         TRUE          FALSE FALSE FALSE  TRUE   FALSE   FALSE    FALSE  TRUE FALSE      FALSE        TRUE   TRUE      TRUE      TRUE   TRUE       FALSE

#   (Intercept) BattingAverage  Runs  Hits Doubles Triples HomeRuns   RBI Walks StrikeOuts StolenBases Errors FreAgtEli FreeAgent ArbEli Arbitration
#5         TRUE          FALSE FALSE  TRUE   FALSE   FALSE    FALSE  TRUE FALSE      FALSE       FALSE  FALSE      TRUE      TRUE   TRUE       FALSE
#7         TRUE          FALSE FALSE  TRUE   FALSE   FALSE    FALSE  TRUE FALSE      FALSE        TRUE   TRUE      TRUE      TRUE   TRUE       FALSE

## NOTE: Both models selected are the same for reduced and full!

lm_log_bic = lm(log(Salary) ~ Hits + RBI + FreAgtEli + FreeAgent + ArbEli, 
                data = baseball)
lm_log_aic = lm(log(Salary) ~ Hits + RBI + StolenBases + Errors + FreAgtEli + FreeAgent + ArbEli, data = baseball)

######################
# AIC model
######################
vif(lm_log_aic)
which.max(vif(lm_log_aic))

lm_log_aic = update(lm_log_aic, . ~ . - Hits)
vif(lm_log_aic)
which.max(vif(lm_log_aic))

######################
# BIC model
######################
vif(lm_log_bic)
which.max(vif(lm_log_bic))

####################################################################
#### Adjusted Models after VIF and final model selection
####################################################################
## After selection lm_red_r2 and lm_full_bic are the same
summary(lm_log_aic)  # Adj R2 Value = 0.8368659
summary(lm_log_bic)  # Adj R2 Value = 0.8413871

adj_r2_sum = c(summary(lm_log_aic)$adj.r.squared, summary(lm_log_bic)$adj.r.squared)
names(adj_r2_sum) <- c("lm_log_aic", "lm_log_bic")

####################################################################
#### Final Model Plots/Tests
####################################################################
## AIC Selected Model
plot(lm_log_aic, which = 1)
plot(lm_log_aic, which = 2)
plot(lm_log_aic, which = 3)
plot(lm_log_aic, which = 4)
plot(lm_log_aic, which = 5)
shapiro.test(lm_log_aic$residuals)

## BIC Selected Model
plot(lm_log_bic, which = 1)
plot(lm_log_bic, which = 2)
plot(lm_log_bic, which = 3)
plot(lm_log_bic, which = 4)
plot(lm_log_bic, which = 5)
shapiro.test(lm_log_bic$residuals)

means = data.frame(rbind(colMeans(baseball[1:17])))
means[14:17] <- round(means[14:17])
aic_pred = exp(predict(lm_log_aic, means, interval = "predict"))
bic_pred = exp(predict(lm_log_bic, means, interval = "predict"))

pred_matrix = matrix(rbind(aic_pred, bic_pred, unt_pred), ncol = length(aic_pred))
colnames(pred_matrix) <- c("fit", "lwr", "upr")
rownames(pred_matrix) <- c("lm_log_aic", "lm_log_bic", "lm_untransformed")

means[14] = 1
bic_fre_agt_eli_pred = exp(predict(lm_log_bic, means, interval = "predict"))
means[15] = 1
bic_fre_agt_and_eli_pred = exp(predict(lm_log_bic, means, interval = "predict"))
means[14] = 0
means[15] = 0
means[16] = 1
bic_arb_eli_pred = exp(predict(lm_log_bic, means, interval = "predict"))
means[16] = 0
bic_fre_agt_eli = exp(predict(lm_log_bic, means, interval = "predict"))


bic_pred_matrix = matrix(rbind(bic_pred, bic_fre_agt_eli_pred, bic_fre_agt_and_eli_pred, bic_arb_eli_pred), ncol = length(aic_pred));
colnames(bic_pred_matrix) <- c("fit", "lwr", "upr")
rownames(bic_pred_matrix) <- c("None", "Free Agent Eligibile", "Free Agent and Eligible", "Arbitration Eligible")
