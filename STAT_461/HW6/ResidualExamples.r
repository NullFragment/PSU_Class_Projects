
n=10
par(mfrow=c(4,3))
for(i in 1:12){
    y=rnorm(n)
    qqnorm(y,pch=20,main=paste("Normal (N=",n,")",sep=""),cex=2)
    qqline(y,lwd=1)
}



n=50
par(mfrow=c(4,3))
for(i in 1:12){
    y=rnorm(n)
    qqnorm(y,pch=20,main=paste("Normal (N=",n,")",sep=""),cex=2)
    qqline(y,lwd=1)
}







n=10
par(mfrow=c(4,3))
for(i in 1:12){
    y=rexp(n)
    qqnorm(y,pch=20,main=paste("Skewed (N=",n,")",sep=""),cex=2)
    qqline(y,lwd=1)
}



n=50
par(mfrow=c(4,3))
for(i in 1:12){
    y=rexp(n)
    qqnorm(y,pch=20,main=paste("Skewed (N=",n,")",sep=""),cex=2)
    qqline(y,lwd=1)
}




n=10
par(mfrow=c(4,3))
for(i in 1:12){
    y=rcauchy(n)
    qqnorm(y,pch=20,main=paste("Heavy-Tailed (N=",n,")",sep=""),cex=2)
    qqline(y,lwd=1)
}



n=50
par(mfrow=c(4,3))
for(i in 1:12){
    y=rcauchy(n)
    qqnorm(y,pch=20,main=paste("Heavy-Tailed (N=",n,")",sep=""),cex=2)
    qqline(y,lwd=1)
}
