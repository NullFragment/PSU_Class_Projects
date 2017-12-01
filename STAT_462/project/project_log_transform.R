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
plot(p_log_f, R2_adj_log_full, xlab = "Number of betas", ylab = "Adjusted R-squared")  # 7 betas / 6 predictors

## Mallow's C
C_p_log_full = RSS_log_full/(sigma_hat_log_full^2) + 2 * p_log_f - n
C_p_diff_log_full = abs(abs(C_p_log_full) - p_log_f)
plot(p_log_f, C_p_log_full, xlab = "Number of betas", ylab = "Mallow's Cp")  # 7 betas / 6 predictors
abline(0, 1)

## AIC
aic_log_full = n * log(RSS_log_full/n) + 2 * p_log_f
plot(p_log_f, aic_log_full, xlab = "Number of betas", ylab = "AIC")  # 7 betas / 6 predictors

## BIC
bic_log_full = n * log(RSS_log_full/n) + p_log_f * log(n)
plot(p_log_f, bic_log_full, xlab = "Number of betas", ylab = "BIC")  # 7 betas / 6 predictors

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
plot(p_log_r, R2_adj_red, xlab = "Number of betas", ylab = "Adjusted R-squared")  # 7 betas / 6 predictors

## Mallow's C
C_p_log_red = RSS_log_red/(sigma_hat_log_full^2) + 2 * p_log_r - n
C_p_diff_log_red = abs(abs(C_p_log_red) - p_log_r)
plot(p_log_r, C_p_log_red, xlab = "Number of betas", ylab = "Mallow's Cp")  # 7 betas / 6 predictors
abline(0, 1)

## AIC
aic_log_red = n * log(RSS_log_red/n) + 2 * p_log_r
plot(p_log_r, aic_log_red, xlab = "Number of betas", ylab = "AIC")  # 7 betas / 6 predictors

## BIC
bic_log_red = n * log(RSS_log_red/n) + p_log_r * log(n)
plot(p_log_r, bic_log_red, xlab = "Number of betas", ylab = "BIC")  # 7 betas / 6 predictors


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
manual_log_both <- c(7, 7, 7, 7)  #Determined for full and reduced models

selection_matrix_log_auto = matrix(rbind(log_full_selection_betas, log_red_selection_betas), 
                                   ncol = length(log_full_selection_betas))
selection_matrix_log_user = matrix(rbind(manual_log_both, manual_log_both), 
                                   ncol = length(manual_log_both))
colnames(selection_matrix_log_auto) <- c("Adj R2", "MallowC", "AIC", "BIC")
rownames(selection_matrix_log_auto) <- c("Full", "Reduced")
colnames(selection_matrix_log_user) <- c("Adj R2", "MallowC", "AIC", "BIC")
rownames(selection_matrix_log_user) <- c("Full", "Reduced")


## NOTE: ALL MODELS BY MANUAL SELECTION ARE THE SAME AS AIC AND BIC.
## Full Manual Selection Results
#   (Intercept) BattingAverage   OBP  Runs  Hits Doubles Triples HomeRuns   RBI Walks StrikeOuts StolenBases Errors FreAgtEli FreeAgent ArbEli Arbitration
#6         TRUE          FALSE FALSE  TRUE FALSE   FALSE   FALSE    FALSE  TRUE FALSE       TRUE       FALSE  FALSE      TRUE      TRUE   TRUE       FALSE

## Reduced (No OBP) Manual Selection Results
#   (Intercept) BattingAverage  Runs  Hits Doubles Triples HomeRuns   RBI Walks StrikeOuts StolenBases Errors FreAgtEli FreeAgent ArbEli Arbitration
#6         TRUE          FALSE  TRUE FALSE   FALSE   FALSE    FALSE  TRUE FALSE       TRUE       FALSE  FALSE      TRUE      TRUE   TRUE       FALSE

lm_log_user = lm(log(Salary) ~ Runs + RBI + StrikeOuts + FreAgtEli + FreeAgent + 
                   ArbEli, data = baseball)

## Models with max R2 to see differences vs manual selection
lm_log_full_r2 = lm(log(Salary) ~ OBP + Runs + Triples + RBI + StrikeOuts + 
                      StolenBases + Errors + FreAgtEli + FreeAgent + ArbEli + Arbitration, 
                    data = baseball)
lm_log_red_r2 = lm(log(Salary) ~ BattingAverage + Runs + Hits + Triples + 
                     RBI + StrikeOuts + StolenBases + Errors + FreAgtEli + FreeAgent + ArbEli, 
                   data = baseball)

######################
# Full model
######################
vif(lm_log_full_r2)
which.max(vif(lm_log_full_r2))

lm_log_full_r2 = update(lm_log_full_r2, . ~ . - Runs)
vif(lm_log_full_r2)
which.max(vif(lm_log_full_r2))

######################
# Reduced model
######################
vif(lm_log_red_r2)
which.max(vif(lm_log_red_r2))

lm_log_red_r2 = update(lm_log_red_r2, . ~ . - Hits)
vif(lm_log_red_r2)
which.max(vif(lm_log_red_r2))

lm_log_red_r2 = update(lm_log_red_r2, . ~ . - Runs)
vif(lm_log_red_r2)
which.max(vif(lm_log_red_r2))

######################
# Manual model
######################
vif(lm_log_user)
which.max(vif(lm_log_user))

lm_log_user = update(lm_log_user, . ~ . - RBI)
vif(lm_log_user)
which.max(vif(lm_log_user))


####################################################################
#### Adjusted Models after VIF and final model selection
####################################################################
## After selection lm_red_r2 and lm_full_bic are the same
summary(lm_log_full_r2)  # Adj R2 Value = 0.7912
summary(lm_log_red_r2)  # Adj R2 Value = 0.7918
summary(lm_log_user)  # Adj R2 Value = 0.7816

adj_r2_sum = c(summary(lm_log_full_r2)$adj.r.squared, summary(lm_log_red_r2)$adj.r.squared, 
               summary(lm_log_user)$adj.r.squared)
names(adj_r2_sum) <- c("lm_log_full_r2", "lm_log_red_r2", "lm_log_user")

#lm_log_full_r2  lm_log_red_r2    lm_log_user 
#     0.7912379      0.7918032      0.7815753 

####################################################################
#### Final Model Plots/Tests
####################################################################
## Full Adj-R^2 Selected Model
plot(lm_log_full_r2, which = 1)
plot(lm_log_full_r2, which = 2)
plot(lm_log_full_r2, which = 3)
plot(lm_log_full_r2, which = 4)
plot(lm_log_full_r2, which = 5)
shapiro.test(lm_log_full_r2$residuals)

## Reduced Adj-R^2 Selected Model
plot(lm_log_red_r2, which = 1)
plot(lm_log_red_r2, which = 2)
plot(lm_log_red_r2, which = 3)
plot(lm_log_red_r2, which = 4)
plot(lm_log_red_r2, which = 5)
shapiro.test(lm_log_red_r2$residuals)

## Manually Selected Model
plot(lm_log_user, which = 1)
plot(lm_log_user, which = 2)
plot(lm_log_user, which = 3)
plot(lm_log_user, which = 4)
plot(lm_log_user, which = 5)
shapiro.test(lm_log_user$residuals)
