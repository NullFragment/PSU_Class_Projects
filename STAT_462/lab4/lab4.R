bear=read.table('bears.txt',header=TRUE,sep='\t')
bear=bear[bear$Obs.No==1,]
y=bear$Weight
X1=as.matrix(cbind(1,bear$Chest.G))
X2=as.matrix(cbind(1,bear$Head.W))
beta1=solve(t(X1)%*%X1,t(X1)%*%y)
beta2=solve(t(X2)%*%X2,t(X2)%*%y)

min_weight_pred_1 = beta1[1] + min(bear$Chest.G)*beta1[2]
min_weight_pred_2 = beta2[1] + min(bear$Head.W)*beta2[2]

n = dim(bear)[1]
p = 2
ybar = mean(bear$Weight)

RSS1 = sum((y - X1%*%beta1)^2)
RSS2 = sum((y - X2%*%beta2)^2)
TSS = sum((y-ybar)^2)

sig1 = sqrt((1/(n-p))*RSS1)
sig2 = sqrt((1/(n-p))*RSS2)

R1 = 1-RSS1/TSS
R2 = 1-RSS2/TSS

beta1
min_weight_pred_1
sig1
R1

beta2
min_weight_pred_2
sig2
R2


plot(bear$Chest.G,bear$Weight)
abline(beta1[1],beta1[2],col='red')

plot(bear$Head.W,bear$Weight)
abline(beta2[1],beta2[2],col='red')