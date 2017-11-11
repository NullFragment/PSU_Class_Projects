####################################################################
#### Setup
####################################################################
## Install libraries (Uncomment to install)
#install.packages('reshape2')
#install.packages('ggplot2')
#install.packages('gridExtra')
#install.packages('leaps')

## Import needed libraries and data
require(reshape2)
require(ggplot2)
require(gridExtra)
require(leaps)

## Load and label data
baseball = read.table("baseball.dat.txt")
names(baseball) <- 
  c(
  'Salary',
  'BattingAverage',
  'OBP',
  'Runs',
  'Hits',
  'Doubles',
  'Triples',
  'HomeRuns',
  'RBI',
  'Walks',
  'StrikeOuts',
  'StolenBases',
  'Errors',
  'FreeAgencyEligibility',
  'FreeAgent',
  'ArbitrationEligibility',
  'Arbitration',
  'Name'
  )
summary(baseball)

## Remove double-counted data
baseball$Hits = baseball$Hits - baseball$Doubles - baseball$Triples - baseball$HomeRuns

## Compute Correlation Matrix
corr_mat <- cor(baseball[,-18])

####################################################################
#### Initial Linear Model Exploration
####################################################################
## Fit inital linear models
lm_salary = lm(Salary ~ Runs + StrikeOuts + StolenBases + OBP, data = baseball)
lm_salary_expanded = lm(Salary ~ Runs + StrikeOuts + StolenBases + Hits + Doubles + Triples + HomeRuns + Walks
                        , data =baseball)
lm_salary_full = lm(Salary ~ . - Name, data = baseball)

## Create Scatter Plots
## Use 1024x576 for image size
subset_obp = baseball[, c('OBP', 'Hits', 'Doubles', 'Triples', 'HomeRuns', 'Walks')]
subset_salary_expanded = baseball[, c('Salary', 'OBP', 'Runs', 'StrikeOuts', 'StolenBases')]

// Salary Model
p1 <- ggplot(baseball, aes(x = OBP, y = Salary)) + geom_point(size = 1)
p2 <- ggplot(baseball, aes(x = Runs, y = Salary)) + geom_point(size = 1)
p3 <- ggplot(baseball, aes(x = StrikeOuts, y = Salary)) + geom_point(size = 1)
p4 <- ggplot(baseball, aes(x = StolenBases, y = Salary)) + geom_point(size = 1)
grid.arrange(p1, p2, p3, p4, ncol = 2)


p1 <- ggplot(baseball, aes(x = Hits, y = Salary)) + geom_point(size = 1)
p2 <- ggplot(baseball, aes(x = Doubles, y = Salary)) + geom_point(size = 1)
p3 <- ggplot(baseball, aes(x = Triples, y = Salary)) + geom_point(size = 1)
p4 <- ggplot(baseball, aes(x = HomeRuns, y = Salary)) + geom_point(size = 1)
p5 <- ggplot(baseball, aes(x = Walks, y = Salary)) + geom_point(size = 1)
grid.arrange(p1, p2, p3, p4, p5, ncol = 3)

## Generate Histograms of data
ggplot(data=melt(baseball[,-18]),aes(x = value)) + facet_wrap(~variable,scales = "free_x") + geom_histogram()


## VIF on OBP Model
X_obp = model.matrix(lm_obp)
R2 = vector("numeric", length(lm_obp$coefficients) - 1)
for (j in 1:length(lm_obp$coefficients) - 1) 
{
  y_tmp = X_obp[, 1 + j]
  x_tmp = as.matrix(X_obp[, -c(1, 1 + j)])
  lm_fit = lm(y_tmp ~ x_tmp)
  R2[j] = summary(lm_fit)$r.squared
}
VIF_obp = 1 / (1 - R2)
names(VIF_obp) = c('Hits', 'Doubles', 'Triples', 'HomeRuns', 'Walks')
VIF_obp

## Free un-necessary memory
rm(p1, p2, p3, p4, p5)
rm(subset_salary_expanded, subset_obp)
rm(j, X_obp, x_tmp, y_tmp, R2)
invisible(gc())

####################################################################
#### Model Selection Exploration
####################################################################
## Use LEAPS for Model Selection Exploration
select_obp = regsubsets(OBP ~ Hits + Doubles + Triples + HomeRuns + Walks,method = "exhaustive",nbest = 1,data = baseball)
select_salary_expanded = regsubsets(Salary ~ OBP + Runs + StrikeOuts + StolenBases,method = "exhaustive",nbest = 1,data = baseball)
select_salary_full = regsubsets(Salary ~ . - Name,method = "exhaustive",nbest = 1,data = baseball)

## Display Data
summary(select_obp)$which
summary(select_salary_expanded)$which
summary(select_salary_full)$which

