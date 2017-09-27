bear=read.table('bears.txt',header=TRUE,sep='\t')
bear=bear[bear$Obs.No==1,]

## Linear System Solvera
lm_six = lm(bear$Weight~bear$Head.L+bear$Head.W+bear$Neck.G+bear$Length+bear$Chest.G)
RSS_six = sum(lm_six$residuals^2)


lm_reduced = lm(bear$Weight~bear$Head.L+bear$Neck.G+bear$Chest.G)
RSS_reduced = sum(lm_reduced$residuals^2)


## Set useful variables
n = dim(bear)[1]
p=length(lm_six$coefficients)
q=length(lm_reduced$coefficients)
f=((RSS_reduced-RSS_six)/(p-q))/(RSS_six/(n-p))
pvalue=pf(f,p-q,n-p,lower.tail=FALSE)


## Display Results
summary(lm_six)
summary(lm_reduced)
RSS_six
RSS_reduced
f
pvalue
anova(lm_reduced,lm_six)
