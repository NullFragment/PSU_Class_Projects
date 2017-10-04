cement=read.table('cement.txt',header=TRUE,sep='\t')

## Linear System Solver
lm_aluminum = lm(Hardness_cement~Aluminum, data=cement)
quantiles = quantile(cement$Aluminum, probs=c(0.5,0.9))
estimates = predict(lm_aluminum, data.frame(Aluminum=quantiles))
confidence = predict(lm_aluminum, data.frame(Aluminum=quantiles), interval="confidence", level=.95)
prediction = predict(lm_aluminum, data.frame(Aluminum=quantiles), interval="prediction", level=.95)
confwidth = cbind(confidence[,3]-confidence[,2])
colnames(confwidth) <- c("CI Width")
predwidth = cbind(prediction[,3]-prediction[,2])
colnames(predwidth) <- c("PI Width")

quantiles
estimates
confidence
prediction
confwidth
predwidth
