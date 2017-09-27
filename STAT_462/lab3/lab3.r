times = read.table('record.txt',header=TRUE,sep=' ')
attach(times)
plot(times)
cor(times)
y = m400
x = m100
x_bar = mean(x)
y_bar = mean(y)
b1 = sum((x-x_bar)*(y-y_bar))/sum((x-x_bar)^2)
b0 = y_bar - b1*x_bar
X=as.matrix(cbind(1,m100))
beta=solve(t(X)%*%X,t(X)%*%y)
beta
plot(times$m100,times$m400)
abline(beta[1],beta[2],col='red')