library(data.table)
library(ggplot2)
library(scales)
library(glmnet)
library(rpart)
library(gbm)
library(randomForest)
library(pscl)
library(gplots)
library(ROCR)
library (SDMTools)
library(caret)
library(e1071)
theme_set(theme_bw())

credit.data <- fread("credit_orig_columns.csv", stringsAsFactors = T)

credit.data[, validation:=0]
credit.data[sample(nrow(credit.data), 10000), validation:=1]
credit.data.validation <- credit.data[validation==1]
credit.data.train <- credit.data[validation==0]

f1 <- as.formula(default.payment.next.month ~ .)
x1.credit.data.train <- model.matrix(f1, credit.data.train)[, -1]
y.credit.data.train <- credit.data.train$default.payment.next.month
x1.validation <- model.matrix(f1, credit.data.validation)[, -1]
y.credit.data.validation<- credit.data.validation$default.payment.next.month





fit.lm <- lm(f1, credit.data.train)
yhat.train.lm <- predict(fit.lm)
rsq.train.lm <- mean((y.credit.data.train - yhat.train.lm)^2)
summary(fit.lm)
rsq.train.lm

yhat.validation.lm <- predict(fit.lm, credit.data.validation, type= 'response')
rsq.validation.lm <- mean((y.credit.data.validation - yhat.validation.lm)^2)
rsq.validation.lm


fit.glm <- glm(f1,family=binomial(link='logit'),data=credit.data.train)
summary(fit.glm)

yhat.train.glm <- predict(fit.glm,newdata = credit.data.train, type= 'response')
rsq.train.glm <- mean((y.credit.data.train - yhat.train.glm)^2)
rsq.train.glm

yhat.validation.glm <- predict(fit.glm,newdata = credit.data.validation, type= 'response')
rsq.validation.glm <- mean((y.credit.data.validation - yhat.validation.glm)^2)
rsq.validation.glm

summary(fit.glm)

anova(fit.glm, test="Chisq")

fit.lasso <- cv.glmnet(x1.credit.data.train, y.credit.data.train, alpha = 1, nfolds = 10)
yhat.train.lasso <- predict(fit.lasso, x1.credit.data.train, s = fit.lasso$lambda.min)
rsq.train.lasso <- mean((y.credit.data.train - yhat.train.lasso)^2)
print(fit.lasso)
coef(fit.lasso, s=0.1)
rsq.train.lasso

yhat.validation.lasso <- predict(fit.lasso, x1.validation, s = fit.lasso$lambda.min)
rsq.validation.lasso <- mean((y.credit.data.validation - yhat.validation.lasso)^2)
rsq.validation.lasso

fit.ridge <- cv.glmnet(x1.credit.data.train, y.credit.data.train, alpha = 0, nfolds = 10)
coef(fit.ridge, s=0.1)
yhat.train.ridge <- predict(fit.ridge, x1.credit.data.train, s = fit.ridge$lambda.min)
rsq.train.ridge <- mean((y.credit.data.train - yhat.train.ridge)^2)
rsq.train.ridge

yhat.validation.ridge <- predict(fit.ridge, x1.validation, s = fit.ridge$lambda.min)
rsq.validation.ridge <- mean((y.credit.data.validation - yhat.validation.ridge)^2)
rsq.validation.ridge

fit.elastic <- cv.glmnet(x1.credit.data.train, y.credit.data.train, alpha = .5, nfolds = 10)
coef(fit.elastic, s=0.01)
yhat.train.elastic <- predict(fit.elastic, x1.credit.data.train, s = fit.elastic$lambda.min)
rsq.train.elastic <- mean((y.credit.data.train - yhat.train.elastic)^2)
rsq.train.elastic

yhat.validation.elastic <- predict(fit.elastic, x1.validation, s = fit.elastic$lambda.min)
rsq.validation.elastic <- mean((y.credit.data.validation - yhat.validation.elastic)^2)
rsq.validation.elastic

fit.tree <- rpart(f1, credit.data.train, control = rpart.control(cp = 0.001))
par(xpd = TRUE)
plot(fit.tree, compress = TRUE)
text(fit.tree, use.n=TRUE)

yhat4.tree <- predict(fit.tree, credit.data.train)
rsq.tree <- mean((yhat4.tree - y.credit.data.train)^2)
rsq.tree

yhat4.tree.validation <- predict(fit.tree, credit.data.validation)
rsq.tree.validation <- mean((yhat4.tree.validation - y.credit.data.validation)^2)
rsq.tree.validation

fit.rndfor <- randomForest(x1.credit.data.train, y.credit.data.train, do.trace = 1, ntree = 50)
yhat.rndfor <- predict(fit.rndfor)
rsq.rndfor <- mean((y.credit.data.train - yhat.rndfor)^2)
rsq.rndfor

yhat.rndfor.validation <- predict(fit.rndfor, credit.data.validation)
rsq.rndfor.validation <- mean((y.credit.data.validation - yhat.rndfor.validation)^2)
rsq.rndfor.validation


fit.btree <- gbm(f1,
                 data = credit.data.train,
                 distribution = "gaussian",
                 n.trees = 100,
                 interaction.depth = 2,
                 shrinkage = 0.1)


yhat.btree <- predict(fit.btree, n.trees = gbm.perf(fit.btree, plot.it = FALSE), type="response")
rsq.btree <- mean((y.credit.data.train - yhat.btree)^2)
rsq.btree

yhat.btree.validation <- predict(fit.btree, credit.data.validation, n.trees = gbm.perf(fit.btree, plot.it = FALSE), type="response")
rsq.btree.validation <- mean((y.credit.data.validation - yhat.btree.validation)^2)
rsq.btree.validation

max(yhat.btree.validation)
mean(yhat.btree.validation)
summary(yhat.btree.validation)


fitted.results <- ifelse(yhat.btree.validation > 0.5,1,0)
confusionMatrix(fitted.results, y.credit.data.validation, positive = "1" , dnn = c("Predicted", "Actual"))

misClasificError <- mean(fitted.results != credit.data.validation$default.payment.next.month)
print(paste('Accuracy',1-misClasificError))


p <- predict(fit.btree, newdata=credit.data.validation, type="response", n.trees = 100)
pr <- prediction(p, credit.data.validation$default.payment.next.month)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


hist(credit.data.validation$default.payment.next.month, ,
     right=TRUE,
     col=c("red","blue"),
     main="Default Rate Actual",
     xlab="Default Y/N?",
     breaks=c(0,0.5,1),
     ylim=c(0,10000),
     xaxt = 'n',
     labels=c("No","Yes")
)

