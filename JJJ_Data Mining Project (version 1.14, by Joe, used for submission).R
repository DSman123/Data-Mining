# Joel, Joe, Jeff
# Data Mining (01:958:588)
# Professor Guo
# April 3, 2021

### Preliminaries
library(ggplot2)
library(ISLR)
library(glmnet)
library(leaps)
library(caret)
library(tidyverse)
library(boot)
library(broom)
library(tidytext)
library(rvest)
library(purrr)
library(gridExtra)
library(grid)
library(lattice)
library(bnpa)
library(plyr) 
library(tree)
library(missForest)
library(class)
library(PerformanceAnalytics)
library(randomForest)
library(gbm)
library(pROC)
library(mgcv)
library(cvTools)
library(gam)

### Cleaning data
## get data
## data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
data_url="C:/Users/koljo/Downloads/crx.data" # The website was not responding 
# so reading from a copy of the data stored locally

data1 = read.csv(data_url, header = FALSE)
data=data1
## Replacing string categorical variables by their levels
data$V1 = str_replace_all(data$V1,c("b"="0","a"="1"))
data$V4 = str_replace_all(data$V4,c("u"="0","y"="1","l"="2","t"="3"))
data$V5 = str_replace_all(data$V5,c("g"="0","p"="1","gg"="2"))
data$V6 = str_replace_all(data$V6,c("c"="0","d"="1","cc"="2","i"="3","j"="4",
                                      "k"="5","m"="6","r"="7","q"="8","w"="9",
                                      "x"="10","e"="11","aa"="12","ff"="13"))
data$V7 = str_replace_all(data$V7,c( "v"="0","h"="1","bb"="2","j"="3","n"="4",
                                       "z"="5","dd"="6","ff"="7","o"="8"))
data$V9 = str_replace_all(data$V9,c("t"="0","f"="1"))
data$V10 = str_replace_all(data$V10,c("t"="0","f"="1"))
data$V12 = str_replace_all(data$V12,c("t"="0","f"="1"))
data$V13 = str_replace_all(data$V13,c("g"="0","p"="1","s"="2"))
data$V16 = str_replace_all(data$V16,c("\\+"="1","\\-"="0"))
## rename columns (must look at dataset description)
names(data) 
names(data) = c("male","age","debt","married","bank_customer","education","ethnicity",
                "years_employed","prior_default","employed","credit_score","drivers_license",
                "citizenship","zip_code","income","approved")
names(data) # now has meaningful name
# To replace the missing values represented by ? by NA
na_func = function(.x) str_replace_all(.x, "\\?","NA")
data = map_df(data, na_func)
check.na(data)
view(data)
# Converting all values to numeric
data = map_df(data,as.numeric) %>% as.data.frame()
# Converting nominal and ordinal categorical variables
data[,c(1,4,5,7,9,10,12,13,16)] = map_df(data[,c(1,4,5,7,9,10,12,13,16)], as.factor)
data$education = factor(data$education, ordered = TRUE)
# Imputing missing values using random forest
set.seed(4521)
imp_data = missForest(data,ntree=200,mtry=10) # To impute missing variables using random forest; run once
data = imp_data$ximp
check.na(data) # should say 0 NAs now
# Performance mesearement using out of bag error
imp_OOBerror = imp_data$OOBerror # can be used to check performance of random forest
# checking levels of categorical variables
levels(data$male) 
levels(data$married) # note: original data has 4 levels, but 1 level = ?s b/c undiscolved info
levels(data$bank_customer)
levels(data$education)
levels(data$ethnicity)
levels(data$prior_default)
levels(data$employed)
levels(data$drivers_license)
levels(data$citizenship)
levels(data$approved) # response variable (y)

### Descriptive Statistics (EDA)
## Summaries
dim(data)
head(data)
summary(data)
## Plots 
# Bar charts (since variables are categorical)
p1 = ggplot(data, aes(x = male)) + 
  geom_bar() + ggtitle("Bar chart of male")
p2 = ggplot(data, aes(x = approved)) +
  geom_bar() + ggtitle("Bar chart of approved") 
p3 = ggplot(data, aes(x = prior_default)) +
  geom_bar() + ggtitle("Bar chart of prior default")
p4 = ggplot(data, aes(x = drivers_license)) +
  geom_bar() + ggtitle("Bar chart of driver's license")
p5 = ggplot(data, aes(x =married)) + 
  geom_bar() + ggtitle("Bar chart of married")
p6 = ggplot(data, aes(x = bank_customer)) + 
  geom_bar() + ggtitle("Bar chart of bank_customer")
p7 = ggplot(data, aes(x = education)) + 
  geom_bar() + ggtitle("Bar chart of education")
p8 = ggplot(data, aes(x = ethnicity)) + 
  geom_bar() + ggtitle("Bar chart of ethnicity")
p9 = ggplot(data, aes(x = employed)) + 
  geom_bar() + ggtitle("Bar chart of employed")
p10 = ggplot(data, aes(x = citizenship)) + 
  geom_bar() + ggtitle("Bar chart of citizenship")
grid.arrange(p1, p2, p3, p4, nrow = 2)
grid.arrange(p5, p6, p7, p8, nrow = 2)
grid.arrange(p9, p10, nrow = 2)# must have ggplot2 & last 3 libraries to make this work
# Histograms (since variables are continuous)
par(mfrow = c(2, 3))
hist(data$age)
hist(data$debt)
hist(data$years_employed)
hist(data$credit_score)
hist(data$zip_code)
hist(data$income)
# Plots (reference: https://r4ds.had.co.nz/exploratory-data-analysis.html)
par(mfrow = c(1, 1)) # must run, resets par()
ggplot(data = data) + # plot b/w 2 categorical variables # Is it only one plot?
  geom_count(mapping = aes(x = prior_default, y = approved))
plot(data$credit_score, data$approved, main = "Scatterplot of Credit Score & Approved", # plot cont. variable against categorical
     xlab = "Credit Score ", ylab = "Approved", pch=19)

### Model building
## splitting and checking data
set.seed(1)
set = sample(1:nrow(data), nrow(data)*0.8) # 0.5 = 1/2  ### Note: TA suggested 80 train/20 test split instead of 50/50
# Note: count(train$married) & count(test$married) are different from 50/50 split!
train = data[set, ] # training set
test = data[-set, ] # test set
approved.train = train$approved
approved.test = test$approved
dim(train) # 552 rows, 16 cols
dim(test) # 138 rows, 16 cols
dim(data) # 690 rows, 16 cols
count(approved.test) # load plyr library first before running this
count(approved.train) # seems to be a nice ratio of both 0's and 1's for both sets
# summary(train$married)
# summary(test$married)

## Logistic regression
# Whole model: Using all the 15 predictors
set.seed(1)
glm.fits.1 = glm(approved ~ ., data = train, family = binomial)
summary(glm.fits.1)

# Two simplified models: Removing the predictors with large p-value
glm.fits.2 = glm(approved ~ . -male -age -debt -bank_customer -education -employed -credit_score,
                 data = train, family = binomial)
summary(glm.fits.2)

glm.fits.3 = glm(approved ~ . -male -age -debt -bank_customer -education -employed -credit_score
                 -ethnicity -drivers_license -years_employed -zip_code, data = train, family = binomial)
summary(glm.fits.3)
# Model comparison: Compare their performance on predicting the testing set
# The AIC values of the three models are very close to each other's
nrow(test) # 552, put in rep()'s
glm.probs.1 = predict(glm.fits.1, test, type = "response") # note: means only have 2 obs in test, but 0 in train for data$married
glm.pred.1 = rep(0, 138)
glm.pred.1[glm.probs.1 > .5] = 1
table(glm.pred.1, test$approved)
mean(glm.pred.1 == test$approved)  # Model 1 correctly predicted 84.35% of the observations in the testing set
mean(glm.pred.1 != test$approved)

glm.probs.2 = predict(glm.fits.2, test, type = "response")
glm.pred.2 = rep(0, 138)
glm.pred.2[glm.probs.2 > .5] = 1
table(glm.pred.2, test$approved)
mean(glm.pred.2 == test$approved)  # Model 2 correctly predicted 84.35% of the observations in the testing set
mean(glm.pred.2 != test$approved) 

glm.probs.3 = predict(glm.fits.3, test, type = "response")
glm.pred.3 = rep(0, 138)
glm.pred.3[glm.probs.3 > .5] = 1
table(glm.pred.3, test$approved)
mean(glm.pred.3 == test$approved)  # Model 3 correctly predicted 84.93% of the observations in the testing set
mean(glm.pred.3 != test$approved)
# Among the three models, model 3 is the simplest and has the best performance.
# Model 3: approved ~ married + prior_default + citizenship + income
# Leave-One-Out Cross-Validation for Logistic
# Fit all the three models with whole data
glm.fits.11 = glm(approved ~ ., data = data, family = binomial)
glm.fits.22 = glm(approved ~ . -male -age -debt -bank_customer -education -employed -credit_score,
                  data = data, family = binomial)
glm.fits.33 = glm(approved ~ . -male -age -debt -bank_customer -education -employed -credit_score
                  -ethnicity -drivers_license -years_employed -zip_code, data = data, family = binomial)
# Obtain Delta
cv.err.11 = cv.glm(data, glm.fits.11)
cv.err.22 = cv.glm(data, glm.fits.22)
cv.err.33 = cv.glm(data, glm.fits.33)
cv.err.11$delta
cv.err.22$delta
cv.err.33$delta

### AUC
glm.fit.3_roc = roc((as.integer(approved.test)-1),glm.pred.3, plot=T,print.auc=T,
                    col="green",lwd=4,legacy.axes=T, main="ROC curves for logistic regression")

# Assumptions check-Linearity between logit and continuous variables  ### if asked about logit assumptions ###
logit = predict(glm(approved ~ ., data = data, family = binomial))
ggplot(data,aes(age,logit)) + geom_smooth() + geom_point()
ggplot(data,aes(debt,logit)) + geom_smooth() + geom_point()
ggplot(data,aes(years_employed,logit)) + geom_smooth() + geom_point()
ggplot(data,aes(credit_score,logit)) + geom_smooth() + geom_point()
ggplot(data,aes(zipcode,logit)) + geom_smooth() + geom_point()
ggplot(data,aes(income,logit)) + geom_smooth() + geom_point()
# To check multicollinearity among continous variables
chart.Correlation(data[,c(2,3,8,11,14,15)] ,method="spearman")

## Classification tree
# building tree (see pg. 324 of textbook for reference)
set.seed(1)
tree.data = tree(approved ~ ., data = train) # tree.data = R gives output corresponding to each branch of tree
tree.pred = predict(tree.data, test, type = "class")
table(tree.pred, approved.test) # evaluating performance on test data
100*round((65 + 50)/138, 4) # 83.33% test observations are correctly classified
100-100*round((65 + 50)/138, 4)
# performing CV on tree (See pg. 326)
set.seed(1)
cv.data = cv.tree(tree.data, FUN = prune.misclass) # let's see if pruning tree leads = improved results
names(cv.data) # cv.tree = determines optimal level of tree complexity
cv.data # dev = corresponds to CV error rate; tree with 2 terminal nodes = has lowest CV error
# plot error rate as function of size & k
par(mfrow = c(1, 2))
plot(cv.data$size, cv.data$dev, type = "b")
plot(cv.data$k, cv.data$dev, type = "b")
par(mfrow = c(1, 1)) # reset par
# pruning & plotting tree of 5 nodes
prune.data = prune.misclass(tree.data, best = 2) # based on cv.data, 2 nodes is best
plot(prune.data)
text(prune.data, pretty = 0)
tree.preds = predict(prune.data, test, type = "class") # check how well pruned tree performs on test set
table(tree.preds, approved.test)
100*round((61 + 55)/138, 4) # 84.06% of test observations are correctly classified
100-100*round((66 + 55)/138, 4)

### AUC
tree_roc = roc((as.integer(approved.test)-1), (as.integer(tree.preds)-1), plot=T,print.auc=T,
                    col="green",lwd=4,legacy.axes=T, main="ROC curves for classification tree")

## KNN
set.seed(1)
# getting data ready for knn classification
knn_x.train = train[,-16] %>% sapply(unclass)%>% as.matrix()
knn_x.test = test[,-16] %>% sapply(unclass)%>% as.matrix()
knn_y.train = train[,16]
knn_y.test = test[,16]
# mean and standard deviation vector for the training data
mean_vec = apply(knn_x.train, 2, mean)
sd_vec = apply(knn_x.train,2,sd)
# standardized test and training predictors data
knn_x_train = scale(knn_x.train,center = mean_vec,
                    scale=sd_vec)
knn_x_test = scale(knn_x.test,center = mean_vec,
                   scale=sd_vec)
# Choosing optimal k and the lowest test error
knn_acc = numeric()
k_vec = c(1:nrow(knn_x_test))
for (i in k_vec){
  knn.pred = class::knn(knn_x_train,knn_x_test,knn_y.train,k=i,use.all = T)
  knn_conf_table = table(knn.pred,knn_y.test)
  knn_acc[i] = mean(knn.pred==knn_y.test)
  append(knn_acc,knn_acc[i])
}
K_acc.df = cbind(k_vec,knn_acc)%>%as.data.frame()
# Optimal k
best_k = which.max(knn_acc)
best_k
# Largest accuracy
largest_acc = K_acc.df[which.max(knn_acc),2]
largest_acc
# Plot of accuracy vs k
plot(k_vec,knn_acc, type = "l" ,col=ifelse(k_vec==best_k, 'red', 'blue'), 
     xlab = "Value of k", ylab = "Accuracy of classification",cex=.3)
title("KNN, Test accuracy vs k")
points(best_k,largest_acc,col=2)


### AUC
opt_knn.pred = class::knn(knn_x_train,knn_x_test,knn_y.train,best_k,use.all = T)
knn_roc = roc((as.integer(approved.test)-1), (as.integer(opt_knn.pred)-1), plot=T,print.auc=T,
               col="green",lwd=4,legacy.axes=T, main="ROC curves for KNN")


##### Random Forest
set.seed(1)
A_vec = rep(1:5,each=6) # to create vector of 1 to 5 each repeated 6 times
B_vec = rep(1:6,5) # to create vector of 1 to 6, 5 times
for (nodesize in 1:5){
      OOB_error500 = numeric() # create empty vector to store OOB error 
      for (nparam in 1:6){
      rf.data = randomForest(approved ~ ., data = train, 
                             nodesize=nodesize, mtry = nparam, importance = TRUE)
      OOB_error500[nparam] = rf.data$err.rate[500,1]
      append(OOB_error500,OOB_error500[nparam])
      }
}
OOB_param =  cbind(A_vec,B_vec,OOB_error500) %>% as.data.frame()
min_OOB_error_index = which.min(OOB_error500) # Find the index of the minimun out 
                                         # of bag error
opt_param = OOB_param[min_OOB_error_index,]
opt_param # Vector of optimal paramtors where A_vec represent the terminal 
          # node size, B_vec the optimal number of variables

# We can build and evaluate the final random forest model with those parameter
final.rf = randomForest(approved ~ ., data = train, 
                        nodesize=OOB_param[min_OOB_error_index,1],
                        mtry = OOB_param[min_OOB_error_index,2], importance = TRUE) 

# The model can be evaluated using the test data
rf.preds = predict(final.rf, test) # prediction using test data
rf.acc=mean(rf.preds== approved.test) # accuracy of model
rf.acc
# AUC
rf_roc = roc((as.integer(approved.test)-1), (as.integer(rf.preds)-1), plot=T,print.auc=T,
              col="green",lwd=4,legacy.axes=T, main="ROC curves for Random Forest")

## GAM
# Fitting model
set.seed(1)
gam_fit = gam(approved ~ male + s(age) + s(debt) + married + bank_customer + education + ethnicity + s(years_employed) + prior_default + employed + credit_score + drivers_license + citizenship + zip_code + s(income), family = binomial, data = train)
# Note: male, age, debt, years_employed, and income = continuous terms, so we're using them for smoothing; others are discrete or categorical.
summary(gam_fit)
layout(matrix(1:4, nrow = 2))
plot(gam_fit, shade = T)

# predict function
gam_pred <- 1 - 1 / (1 + exp(predict(gam_fit, test))) > 0.5
table(gam_pred, approved.test) # evaluating performance on test data
mean(gam_pred == as.numeric(approved.test) - 1)*100 # gives 85.51% test accuracy

## Trying degree n>1 polynomial terms

## Adding polynomial degree 2 for age
gam_fit1 = gam(approved ~ male + s(age,2) + s(debt) + married + bank_customer + education + ethnicity + s(years_employed) + prior_default + employed + credit_score + drivers_license + citizenship + zip_code + s(income), family = binomial, data = train)

# predict function
gam_pred1 <- 1 - 1 / (1 + exp(predict(gam_fit1, test))) > 0.5
table(gam_pred1, approved.test) # evaluating performance on test data
mean(gam_pred1 == as.numeric(approved.test) - 1)*100 # gives 84.78% test accuracy

## Adding polynomial degree 2 for debt
gam_fit2 = gam(approved ~ male + s(age) + s(debt,2) + married + bank_customer + education + ethnicity + s(years_employed) + prior_default + employed + credit_score + drivers_license + citizenship + zip_code + s(income), family = binomial, data = train)

# predict function
gam_pred2 <- 1 - 1 / (1 + exp(predict(gam_fit2, test))) > 0.5
table(gam_pred2, approved.test) # evaluating performance on test data
mean(gam_pred2 == as.numeric(approved.test) - 1)*100 # gives 83.33% test accuracy

## Adding polynomial degree 2 for years_employed
gam_fit3 = gam(approved ~ male + s(age) + s(debt) + married + bank_customer + education + ethnicity + s(years_employed,2) + prior_default + employed + credit_score + drivers_license + citizenship + zip_code + s(income), family = binomial, data = train)

# predict function
gam_pred3 <- 1 - 1 / (1 + exp(predict(gam_fit3, test))) > 0.5
table(gam_pred3, approved.test) # evaluating performance on test data
mean(gam_pred3 == as.numeric(approved.test) - 1)*100 # gives 84.78% test accuracy

## Adding polynomial degree 2 for age
gam_fit4 = gam(approved ~ male + s(age) + s(debt) + married + bank_customer + education + ethnicity + s(years_employed) + prior_default + employed + credit_score + drivers_license + citizenship + zip_code + s(income,2), family = binomial, data = train)

# predict function
gam_pred4 <- 1 - 1 / (1 + exp(predict(gam_fit4, test))) > 0.5
table(gam_pred4, approved.test) # evaluating performance on test data
mean(gam_pred4 == as.numeric(approved.test) - 1)*100 # gives 85.51% test accuracy

# Comparing the test accuracy of the five Gam models, it appears that there is 
# no need to add a polynomial term in the model, so model gam_fit is retained, its accuracy:

mean(gam_pred == as.numeric(approved.test) - 1)*100 # gives 85.51% test accuracy


# AUC 
par(mfrow = c(1, 1))
gam_roc = roc((as.numeric(approved.test)-1),as.numeric(gam_pred), plot=T,print.auc=T,
              col="green",lwd=4,legacy.axes=T, main="ROC curves for GAM")


##### Boosting
# The range of parameters is reduced to make computation less intensive
set.seed(1)
ntrees = 1000 # max number of trees
shr_val = c(0.001, 0.1,0.2,0.3,0.4) # vector of shrinkage parameters
depth_val = seq(1,10,1)  # vector of depth values

# The following two variables are used in storing model accuracy
depth_vec = rep(depth_val,each=length(shr_val))
shr_vec = rep(shr_val, length(depth_val))

best_iter_vec = numeric()
cv_error_vec = numeric()
i=0 
for (depth in depth_val) {
  for (lambda in shr_val) {
    boost.model = gbm((as.numeric(approved)-1) ~ ., data = train, distribution = "bernoulli",
                      n.trees = ntrees, train.fraction = 0.7,interaction.depth = depth,
                      shrinkage = lambda, cv.folds = 5, verbose = F)
    i=i+1 # Step check
    print(i)
    best_iter = gbm.perf(boost.model,plot.it = F ,method = "cv")
    best_iter_vec = append(best_iter_vec,best_iter )    
    cv_error = min(boost.model$cv.error)
    cv_error_vec = append(cv_error_vec, cv_error)   
    
  }
}
boost_param =  cbind(shr_vec,depth_vec,best_iter_vec,cv_error_vec) %>% as.data.frame()

boost.opt = boost_param[which.min(cv_error_vec),] # To output optimal parameters

# Now we have the optimal parameters to build the model

boost_opt.model = gbm((as.numeric(approved)-1) ~ ., data = train, distribution = "bernoulli",
                      n.trees = as.numeric(boost.opt[3]), train.fraction = 0.7,interaction.depth = as.numeric(boost.opt[2]),
                      shrinkage = as.numeric(boost.opt[1]), cv.folds = 5, verbose = F)

# Because we need to specify a cutoff value for probabilities, let's find the optimal cutoff
# Finding best cutoff
cutoff = seq(0.1,0.9,0.1) # to find the best cutoff value for probabilities

boost_acc_cutoff = numeric()
for (cut_at in cutoff){
  boost.probs_cutoff = predict(boost_opt.model, test, n.trees = as.numeric(boost.opt[3]),type = "response")
  boost.preds_cutoff = ifelse(boost.probs_cutoff > cut_at , 1, 0)
  boost_acc_cut = mean(boost.preds_cutoff == (as.numeric(approved.test)-1))
  boost_acc_cutoff=append(boost_acc_cutoff , boost_acc_cut)
}

preds_cutoff =  cbind(cutoff,boost_acc_cutoff) %>% as.data.frame()
boost.opt_cutoff = preds_cutoff[which.max(boost_acc_cutoff),] # To output optimal cutoff and acc

# Final prediction
best.preds_cutoff = ifelse(boost.probs_cutoff > 0.5 , 1, 0)
test_acc = mean(best.preds_cutoff==(as.numeric(test$approved)-1))
test_acc

# AUC
# Prediction according to the best probability cutoff point

boost_roc = roc((as.numeric(test$approved)-1), best.preds_cutoff, plot=T,print.auc=T,
                col="green",lwd=4,legacy.axes=T, main="ROC curves for Boosting")


