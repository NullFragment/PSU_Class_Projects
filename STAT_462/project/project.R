baseball = read.table("baseball.dat.txt")
names(baseball) <- c('Salary','BattingAverage','OBP','Runs',
                     'Hits','Doubles','Triples','HomeRuns',
                     'RBI','Walks','Strike-outs','StolenBases','Errors',
                     'FreeAgencyEligibility','FreeAgent','ArbitrationEligibility','Arbitration','Name')

Multilm <- lm(V1~.-V18-V17-V16-V15-V14, data=baseball)
summary(Multilm)
names(Multilm$coefficients) <- c("Intercept","Bat Avg","OBP","Runs","Hits"
                                 ,"Double","Triple","HomeRuns","RBI","Walks","Strikes",
                                 "Stolen","Errors")


X=model.matrix(Multilm)
R2=vector("numeric",12)
for(j in 1:12){
  y_tmp=X[,1+j]
  x_tmp=as.matrix(X[,-c(1,1+j)])
  lm_fit=lm(y_tmp~x_tmp)
  R2[j]=summary(lm_fit)$r.squared
}
VIF=1/(1-R2)
#names(VIF)=c('x1','x2','x3')
VIF

