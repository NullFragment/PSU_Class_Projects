lm_forward1 = update(lm_forward, . ~ . + NeckC)
d = summary(lm_forward1)$coefficients[length(summary(lm_forward1)$coefficients[,4]),4]

lm_forward1 = update(lm_forward, . ~ . + ChestC)
e = summary(lm_forward1)$coefficients[length(summary(lm_forward1)$coefficients[,4]),4]

lm_forward1 = update(lm_forward, . ~ . + HipC)
f = summary(lm_forward1)$coefficients[length(summary(lm_forward1)$coefficients[,4]),4]

lm_forward1 = update(lm_forward, . ~ . + ThighC)
g = summary(lm_forward1)$coefficients[length(summary(lm_forward1)$coefficients[,4]),4]

lm_forward1 = update(lm_forward, . ~ . + KneeC)
h = summary(lm_forward1)$coefficients[length(summary(lm_forward1)$coefficients[,4]),4]

lm_forward1 = update(lm_forward, . ~ . + AnkleC)
i = summary(lm_forward1)$coefficients[length(summary(lm_forward1)$coefficients[,4]),4]

lm_forward1 = update(lm_forward, . ~ . + BicepsC)
j = summary(lm_forward1)$coefficients[length(summary(lm_forward1)$coefficients[,4]),4]

lm_forward1 = update(lm_forward, . ~ . + ForearmC)
k = summary(lm_forward1)$coefficients[length(summary(lm_forward1)$coefficients[,4]),4]

remaining = c(d,e,f,g,h,i,j,k)
min(remaining)
which.min(remaining)
