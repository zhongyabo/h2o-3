setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")
library(Hmisc)
library(foreign)
library(MASS)
library(reshape2)

glmOrdinal <- function() {
  browser()
  carsdata <- read.csv(locate("smalldata/glm_ordinal_logit/car_nidhi.csv"), header=T, stringsAsFactors=F)
  carh2o <- as.h2o(carsdata)
  
  # reorder column levels
  carsdata$buying <- factor(carsdata$buying, levels=c("low", "med", "high", "vhigh"), ordered=TRUE)
  carsdata$maint <- factor(carsdata$maint, levels=c("low", "med", "high", "vhigh"), ordered=TRUE)
  carsdata$doors <- factor(carsdata$doors, levels=c("2", "3", "4", "5more"), ordered=TRUE)
  carsdata$persons <- factor(carsdata$persons, levels=c("2", "4", "more"), ordered=TRUE)
  carsdata$lug_boot <- factor(carsdata$lug_boot, levels=c("small", "med", "big"), ordered=TRUE)
  carsdata$safety <- factor(carsdata$safety, levels=c("low", "med", "high"), ordered=TRUE)
  carsdata$class <- factor(carsdata$class, levels=c("unacc", "acc", "good", "vgood"), ordered=TRUE)
  set.seed(100)
  trainingRows <- sample(1:nrow(carsdata), 0.7 * nrow(carsdata))
  trainingData <- carsdata[trainingRows, ]
  testData <- carsdata[-trainingRows, ]

  options(contrasts = c("contr.treatment", "contr.poly"))
  polrMod <- polr(class ~ safety + lug_boot + doors + buying + maint, data=trainingData, Hess=FALSE)
  #summary(polrMod)
  predictedClass <- predict(polrMod, testData)  # predict the classes directly
  predictedScores <- predict(polrMod, testData, type="p")  # predict the probabilites
  head(predictedScores)
  
  table(testData$class, predictedClass)  # confusion matrix
  mean(as.character(testData$class) != as.character(predictedClass)) 
  
  carh2o$buying <- h2o.asfactor(carh2o$buying)
  h2o.setLevels(carh2o$buying, c("low", "med", "high", "vhigh"))
  carh2o$buying <- h2o.numeric(carh2o$buying)
  carh2o$main <- h2o.asfactor(carh2o$main)
  h2o.setLevels(carh2o$main, c("low", "med", "high", "vhigh"))
  carh2o$main <- h2o.asnumeric(carh2o$main)
  carh2o$doors <- h2o.asfactor(carh2o$doors)
  h2o.setLevels(carh2o$doors, c("2", "4", "more"))
  carh2o$doors <- h2o.asnumeric(carh2o$doors)
  carh2o$lug_bool <- h2o.asfactor(carh2o$lug_bool)
  h2o.setLevels(ccarh2o$lug_bool,c("small", "med", "big"))
  carh2o$lug_bool <- h2o.asnumeric(carh2o$lug_bool)
  carh2o$safety <- h2o.asfactor(carh2o$safety)
  h2o.setLevels(carh2o$safety, c("low", "med", "high"))
  carh2o$safety <- h2o.asnumeric(carh2o$safety)
  carh2o$class <- h2o.asfactor(carh2o$class)
  h2o.setLevels(carh2o$class, c("unacc", "acc", "good", "vgood"))
  h2ocartrain <- carh2o[trainingRows,]
  h2ocartest <- carh2o[-trainingRows,]
  
  X   <- c("safety", "lug_boot", "doors", "buying", "maint")  
  Y<-"class"
  
  Log.info("Build the model")
  m1 <- h2o.glm(y = Y, x = X, training_frame = h2ocartrain, family = "ordinal", beta_epsilon=1e-8, objective_epsilon=1e-10, obj_reg=2/h2o.nrow(D), max_iterations=1000)  
  predh2o = h2o.predict(m1,h2ocartest)
}

doTest("GLM: Ordinal with Car data", glmOrdinal)
