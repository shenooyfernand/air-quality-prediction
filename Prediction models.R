# ============================================
# Logistic Regression
# ============================================

log_model <- multinom(Air.Quality ~ ., data = train_bal)

log_pred <- predict(log_model, xtest)

confusionMatrix(log_pred, ytest)

# ============================================
# SVM
# ============================================

library(e1071)

svm_model <- svm(xtrain, ytrain, kernel="radial")

svm_pred <- predict(svm_model, xtest)

confusionMatrix(svm_pred, ytest)

# ============================================
# KNN
# ============================================

library(class)

knn_pred <- knn(train=xtrain, test=xtest, cl=ytrain, k=5)

confusionMatrix(knn_pred, ytest)

# ============================================
# Neural Network
# ============================================

library(nnet)

nn_model <- nnet(Air.Quality ~ ., data=train_bal, size=5, maxit=200)

nn_pred <- predict(nn_model, xtest, type="class")

confusionMatrix(nn_pred, ytest)