bear=read.table('bears.txt',header=TRUE,sep='\t')
bear=bear[bear$Obs.No==1,]

## Set useful variables
n = dim(bear)[1]
p = 2
ybar = mean(bear$Weight)
y=bear$Weight
X1=as.matrix(cbind(1,bear$Neck.G,bear$Head.W))

## Solve for linear coefficients
beta1=solve(t(X1)%*%X1,t(X1)%*%y)

## Compute residual sum of squares and total sum of squares
RSS1 = sum((y - X1%*%beta1)^2)
TSS = sum((y-ybar)^2)
SS_reg = sum((X1%*%beta1 - ybar)^2)

## Compute variance
variance = (1/(n-p))*RSS1
std_dev = sqrt(variance)

## Compute variance of errors
inv = solve(t(X1)%*%X1)
var_error = sqrt(variance*diag(inv))

## Perform T statistic test
t = beta1/var_error
pvalue=2*pt(abs(t),df=n-p,lower.tail=FALSE)

## Compute R
R1 = 1-RSS1/TSS

## Display Values
TSS - (RSS1 + SS_reg)
var_error
t
pvalue




## Linear System Solver
lm_neck = lm(bear$Weight~bear$Neck.G+bear$Head.W)
summary(lm_neck)