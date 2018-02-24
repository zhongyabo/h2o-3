setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")
library(Hmisc)
library(foreign)
library(MASS)
library(reshape2)

glmOrdinal <- function() {
  browser()
  D <- h2o.uploadFile(locate("smalldata/glm_ordinal_logit/ordinal_nidhi_small.csv"), destination_frame="covtype.hex")  
 # D$apply <- h2o.ifelse(D$apply == "unlikely", 0, h2o.ifelse(D$apply == "somewhat likely", 1, 2)) # reset levels from Megan Kurba
  D$apply <- h2o.asfactor(D$apply)
  h2o.setLevels(D$apply, c("unlikely", "somewhaat likely", "very likely"))
  D$pared <- as.factor(D$pared)
  D$public <- as.factor(D$public)
  
  X   <- c("pared", "public", "gpa")  
  Y<-"apply"
  
  Log.info("Build the model")
  m1 <- h2o.glm(y = Y, x = X, training_frame = D, family = "ordinal", beta_epsilon=1e-8, objective_epsilon=1e-10, max_iterations=1000)  
  predh2o = h2o.predict(m1,D)
  dat <- as.data.frame(D)
  dat$apply <- factor(dat$apply, levels=c("unlikely", "somewhaat likely", "very likely"), ordered=TRUE)
  m <- polr(apply ~ pared + public + gpa, data = dat, Hess=FALSE)
  predictedClassR <- predict(m, dat)
  rPred <- predict(m1, dat, type="p")
  table(dat$apply, predictedClassR)
  summary(m)
}

doTest("GLM: Ordinal with data found by Nidhi at https://stats.idre.ucla.edu/stat/data/ologit.dta", glmOrdinal)
