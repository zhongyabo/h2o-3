setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")
library(Hmisc)
library(foreign)
library(MASS)
library(reshape2)

glmOrdinal <- function() {
  browser()
  D <- h2o.uploadFile(locate("smalldata/glm_ordinal_logit/ordinal_nidhi_small.csv"), destination_frame="covtype.hex")  
  D$apply <- h2o.ifelse(D$apply == "unlikely", 0, h2o.ifelse(D$apply == "somewhat likely", 1, 2)) # reset levels from Megan Kurba
  D$apply <- h2o.asfactor(D$apply)
  #h2o.setLevels(D$apply, c("unlikely", "somewhat likely", "very likely"))
  D$pared <- as.factor(D$pared)
  D$public <- as.factor(D$public)
  
  X   <- c("pared", "public", "gpa")  
  Y<-"apply"
  
  Log.info("Build the model")
  objreg = 1/h2o.nrow(D)
  objregs = c(objreg, objreg/10)#, objreg/100, objreg/1000, objreg/10000, objreg/100000)
  lambdaR = c(0)#,objreg, objreg/10, objreg/100, objreg/1000, objreg/10000, objreg/100000)
  betaL = c(0)#,0.5,1)#,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
  for (lambdaR in lambdaR) {
    for (beta in betaL) {
  for (objR in objregs) {
    m1 <- h2o.glm(y = Y, x = X, training_frame = D, lambda=c(lambdaR), alpha=c(beta), family = "ordinal", beta_epsilon=1e-8, 
                objective_epsilon=1e-10, obj_reg=objR,seed=5338408227998695272)  
    predh2o = as.data.frame(h2o.predict(m1,D))
    Ddata <- as.data.frame(D)
    print(table(Ddata$apply, predh2o$predict))
  }
    }
  }
  D2 <- h2o.uploadFile(locate("smalldata/glm_ordinal_logit/ordinal_nidhi_small.csv"), destination_frame="covtype.hex")  
  dat <- as.data.frame(D2)
  dat$apply <- factor(dat$apply, levels=c("unlikely", "somewhat likely", "very likely"), ordered=TRUE)
  m <- polr(apply ~ pared + public + gpa, data = dat, Hess=FALSE)
  predictedClassR <- predict(m, dat)
  rPred <- predict(m, dat, type="p")
  print(table(dat$apply, predictedClassR))
 # browser()
  #summary(m)
}

doTest("GLM: Ordinal with data found by Nidhi at https://stats.idre.ucla.edu/stat/data/ologit.dta", glmOrdinal)
