rm(list=ls())
library(caret)
library(smotefamily)
library(randomForest)
library(e1071)

set.seed(123)
train <- read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Project 1/Air pollution/train.csv")
test <- read.csv("C:/Users/sheno/OneDrive/Desktop/St4052/Project 1/Air pollution/test.csv")

train <- train[-1]  # remove ID
test <- test[-1]

# Convert target Air.Quality to numeric classes 1,2,3,4
label_map <- c("Good"=1, "Moderate"=2, "Poor"=3, "Hazardous"=4)
train$Air.Quality <- as.numeric(unname(label_map[train$Air.Quality]))
test$Air.Quality <- as.numeric(unname(label_map[test$Air.Quality]))

# Scale numeric features (exclude target)
numeric_cols <- setdiff(names(train), "Air.Quality")
train_mean <- sapply(train[, numeric_cols], mean)
train_sd <- sapply(train[, numeric_cols], sd)

train_scaled <- scale(train[, numeric_cols], center = train_mean, scale = train_sd)
test_scaled <- scale(test[, numeric_cols], center = train_mean, scale = train_sd)

# Combine scaled data with target numeric vector
train_scaled_df <- as.data.frame(train_scaled)
train_scaled_df$Air.Quality <- train$Air.Quality

test_scaled_df <- as.data.frame(test_scaled)
test_scaled_df$Air.Quality <- test$Air.Quality

# Apply SMOTE from smotefamily
smote_output <- SMOTE(train_scaled_df[, numeric_cols], train_scaled_df$Air.Quality, K=5, dup_size=3)
smote_output1 <- SMOTE(smote_output$data[, numeric_cols], smote_output$data$class, K=5, dup_size=1)

# Extract new balanced data
train_balanced <- smote_output1$data
colnames(train_balanced)[ncol(train_balanced)] <- "Air.Quality"

# Convert Air.Quality back to factor with original labels
inv_label_map <- c("1"="Good", "2"="Moderate", "3"="Poor", "4"="Hazardous")
train_balanced$Air.Quality <- factor(inv_label_map[as.character(train_balanced$Air.Quality)],
                                     levels = c("Good", "Moderate", "Poor", "Hazardous"))

# Prepare train and test sets for modeling
xtrain <- train_balanced[, numeric_cols]
ytrain <- train_balanced$Air.Quality

xtest <- test_scaled_df[, numeric_cols]
ytest <- factor(inv_label_map[as.character(test_scaled_df$Air.Quality)],
                levels = c("Good", "Moderate", "Poor", "Hazardous"))

# Set CV control
ctrl <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  classProbs = TRUE,
  savePredictions = "final"
)

# Train Random Forest
set.seed(123)
rf_model <- train(
  x = xtrain,
  y = ytrain,
  method = "ranger",   
  trControl = ctrl,
  tuneGrid = expand.grid(
    mtry = c(1,2, 3, 4),
    splitrule = "gini",         
    min.node.size = 50  
  ),
  num.trees = 20,              # number of trees
  importance = "impurity"
)


# Predict & Evaluate on test
rf_pred <- predict(rf_model, xtest)
cat("Test Accuracy:\n")
print(confusionMatrix(rf_pred, ytest))

# Predict & Evaluate on train
train_pred <- predict(rf_model, xtrain)
cat("\nTrain Accuracy:\n")
print(confusionMatrix(train_pred, ytrain))

# ---- F1 for TEST ----
cm_test <- confusionMatrix(rf_pred, ytest)
byClass_test <- cm_test$byClass

# Handle case if only 1 class is predicted (caret gives a vector instead of matrix)
if (is.null(dim(byClass_test))) {
  f1_test <- 2 * (byClass_test["Sensitivity"] * byClass_test["Pos Pred Value"]) / 
    (byClass_test["Sensitivity"] + byClass_test["Pos Pred Value"])
} else {
  f1_test <- 2 * (byClass_test[,"Sensitivity"] * byClass_test[,"Pos Pred Value"]) / 
    (byClass_test[,"Sensitivity"] + byClass_test[,"Pos Pred Value"])
  names(f1_test) <- rownames(byClass_test)
}

cat("F1-score (Test):\n")
print(f1_test)

# ---- F1 for TRAIN ----
cm_train <- confusionMatrix(train_pred, ytrain)
byClass_train <- cm_train$byClass

if (is.null(dim(byClass_train))) {
  f1_train <- 2 * (byClass_train["Sensitivity"] * byClass_train["Pos Pred Value"]) / 
    (byClass_train["Sensitivity"] + byClass_train["Pos Pred Value"])
} else {
  f1_train <- 2 * (byClass_train[,"Sensitivity"] * byClass_train[,"Pos Pred Value"]) / 
    (byClass_train[,"Sensitivity"] + byClass_train[,"Pos Pred Value"])
  names(f1_train) <- rownames(byClass_train)
}

cat("\nF1-score (Train):\n")
print(f1_train)

# Extract variable importance
importance_vals <- varImp(rf_model)

cat("\nFeature Importance:\n")
print(importance_vals)

library(vip)
plot(importance_vals, top = 10, main = "Top 10 Important Features")

vip(rf_model$finalModel, 
    num_features = 9,       # Show all 9 features
    geom = "col",           # Bar chart style
    aesthetics = list(fill = "steelblue")) +
  theme_minimal() +
  labs(title = "Variable Importance (Random Forest)", x = "Importance", y = "Feature")




library(pdp)
library(ggplot2)

# scaled train data (without target)
train_data <- train_balanced[, numeric_cols]

# Means and SDs used for scaling
means <- train_mean
sds <- train_sd


plot_pdp_original_scale <- function(var_name, model, data, means, sds) {
  # Get PDP for Hazardous class
  pd <- partial(model,
                pred.var = var_name,
                prob = TRUE,
                which.class = "Hazardous",
                train = data)
  
  pd[[paste0(var_name, "_original")]] <- pd[[var_name]] * sds[var_name] + means[var_name]
  
  # Plot PDP with original scale
  p <- ggplot(pd, aes_string(x = paste0(var_name, "_original"), y = "yhat")) +
    geom_line(color = "blue") +
    labs(title = paste("Partial Dependence of", var_name, "on Hazardous Class"),
         x = paste(var_name, "(Original Scale)"),
         y = "Predicted Probability") +
    theme_minimal()
  
  print(p)
}

# Loop over all numeric predictors and plot PDPs
for (v in numeric_cols) {
  plot_pdp_original_scale(v, rf_model$finalModel, train_data, means, sds)
}


