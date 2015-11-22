---
title: "Machine Learning Course Project"
author: "Savir Huitron"
date: "November 22, 2015"
output: html_document
---

## Summary

The main goal of this project is to predict the manner in which people exercise,for the information compiled in the using of devices such as *Nike Fuel Band*. this paper relates how does build the prediction model.


## Preparing the Data

We Separate our data into training and test: 



```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.3
```

```r
ptrain <- read.csv("pml-training.csv")
```

```
## Warning in file(file, "rt"): no fue posible abrir el archivo 'pml-
## training.csv': No such file or directory
```

```
## Error in file(file, "rt"): no se puede abrir la conexión
```

```r
ptest <- read.csv("pml-testing.csv")
```

```
## Warning in file(file, "rt"): no fue posible abrir el archivo 'pml-
## testing.csv': No such file or directory
```

```
## Error in file(file, "rt"): no se puede abrir la conexión
```

We divide in two parts our trainig set:


```r
set.seed(123)

inTrain <- createDataPartition(y=ptrain$classe, p=0.7, list=F)
```

```
## Error in createDataPartition(y = ptrain$classe, p = 0.7, list = F): objeto 'ptrain' no encontrado
```

```r
ptrain1 <- ptrain[inTrain, ]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain' no encontrado
```

```r
ptrain2 <- ptrain[-inTrain, ]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain' no encontrado
```

We clean our data sets removing the missing data, data that doens't work for our prediction model 


```r
# removing the zeros
rz <- nearZeroVar(ptrain1)
```

```
## Error in is.vector(x): objeto 'ptrain1' no encontrado
```

```r
ptrain1 <- ptrain1[, -rz]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain1' no encontrado
```

```r
ptrain2 <- ptrain2[, -rz]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain2' no encontrado
```

```r
#removing the NA's
n_a <- sapply(ptrain1, function(x) mean(is.na(x))) > 0.95
```

```
## Error in lapply(X = X, FUN = FUN, ...): objeto 'ptrain1' no encontrado
```

```r
ptrain1 <- ptrain1[, n_a==F]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain1' no encontrado
```

```r
ptrain2 <- ptrain2[, n_a==F]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain2' no encontrado
```

```r
#this variables are variables that doesn't work for our prediction model

ptrain1 <- ptrain1[, -(1:5)]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain1' no encontrado
```

```r
ptrain2 <- ptrain2[, -(1:5)]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain2' no encontrado
```


## Modeling

As we can remember in our class, the most used algorithms for predictive models are Random Forest and Decision Trees, both are good models, so we can run the first one and make some test for the performance.



```r
fitcontrol <- trainControl(method = "cv", number = 3, verboseIter = F)

# fit model on ptrain1

fit <- train(classe ~ ., data = ptrain1, method = "rf", trControl = fitcontrol)
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain1' no encontrado
```

```r
# summary of the Model 

fit$finalModel
```

```
## Error in eval(expr, envir, enclos): objeto 'fit' no encontrado
```

as we can notice we use 500 trees and 27 splits.

## Evaluation

Now, we use the fitted model to predict the label (“classe”) in ptrain2, and show the confusion matrix to compare the predicted versus the actual labels:



```r
# use model to predict classe in validation set (ptrain2)
preds <- predict(fit, newdata=ptrain2)
```

```
## Error in predict(fit, newdata = ptrain2): objeto 'fit' no encontrado
```



```r
# show confusion matrix to get estimate of out-of-sample error
confusionMatrix(ptrain2$classe, preds)
```

```
## Error in confusionMatrix(ptrain2$classe, preds): objeto 'ptrain2' no encontrado
```


The accuracy under this model is 99.8%, the predicted accuracy for the out-of-sample error is 0.2%.

This a very good result, so this is a good algorithm we can use for the predict on the test.


## Training the Model


What we have done after this line is create a train model for a subsetting data test, and the model has work, in this lines we do the same for the original data set.


```r
# remove variables with nearly zero variance
r_v <- nearZeroVar(ptrain)
```

```
## Error in is.vector(x): objeto 'ptrain' no encontrado
```

```r
ptrain <- ptrain[, -r_v]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain' no encontrado
```

```r
ptest <- ptest[, -r_v]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptest' no encontrado
```

```r
# remove variables that are almost always NA
N_A <- sapply(ptrain, function(x) mean(is.na(x))) > 0.95
```

```
## Error in lapply(X = X, FUN = FUN, ...): objeto 'ptrain' no encontrado
```

```r
ptrain <- ptrain[, N_A == F]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain' no encontrado
```

```r
ptest <- ptest[, N_A == F]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptest' no encontrado
```

```r
# remove the first five variables 

ptrain <- ptrain[, -(1:5)]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain' no encontrado
```

```r
ptest <- ptest[, -(1:5)]
```

```
## Error in eval(expr, envir, enclos): objeto 'ptest' no encontrado
```

```r
# re-fit model using full training set (ptrain)
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
fit <- train(classe ~ ., data=ptrain, method="rf", trControl=fitControl)
```

```
## Error in eval(expr, envir, enclos): objeto 'ptrain' no encontrado
```


## Predictions


```r
# predict on test set
preds <- predict(fit, newdata=ptest)
```

```
## Error in predict(fit, newdata = ptest): objeto 'fit' no encontrado
```

```r
# convert predictions to character vector
preds <- as.character(preds)
```

```
## Error in eval(expr, envir, enclos): objeto 'preds' no encontrado
```

```r
# create function to write predictions to files
pml_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    }
}

# create prediction files to submit
pml_write_files(preds)
```

```
## Error in pml_write_files(preds): objeto 'preds' no encontrado
```
