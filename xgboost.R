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

be=list()
g=numeric(0)
gtest=numeric(0)
gtrain=numeric(0)
for (i in 1:2){   
  # Doing XGBoost for classification purposes.
  set.seed(123)  # For reproducibility
  
  grid_tune <- expand.grid(
    nrounds = 126,           # Try 3 values around your target
    max_depth = 2,                        # Tree depth 1 to 5
    eta = 0.1,               # Learning rates
    gamma = 1,                        # Try with and without pruning
    colsample_bytree =0.6,         # Feature sampling
    min_child_weight = 1,             # Control overfitting
    subsample = 0.8            # Sampling ratio
  )
  
  
  train_control <- trainControl(method = "cv",
                                number=3,
                                search = "random",
                                verboseIter = TRUE,
                                allowParallel = TRUE)
  xgb_tune <- train(x = xtrain,
                    y = ytrain,
                    trControl = train_control,
                    tuneGrid = grid_tune,
                    method= "xgbTree",
                    verbose = TRUE)
  xgb_tune
  
  # Best tune
  xgb_tune$bestTune
  
  # Writing out the best model.
  
  train_control <- trainControl(method = "none",
                                verboseIter = TRUE,
                                allowParallel = TRUE)
  final_grid <- expand.grid(nrounds = xgb_tune$bestTune$nrounds,
                            eta = xgb_tune$bestTune$eta,
                            max_depth = xgb_tune$bestTune$max_depth,
                            gamma = xgb_tune$bestTune$gamma,
                            colsample_bytree = xgb_tune$bestTune$colsample_bytree,
                            min_child_weight = xgb_tune$bestTune$min_child_weight,
                            subsample = xgb_tune$bestTune$subsample)
  xgb_model <- train(x = xtrain,
                     y = ytrain,
                     trControl = train_control,
                     tuneGrid = final_grid,
                     method = "xgbTree",
                     verbose = TRUE)
  
  predict(xgb_model, xtest)
  
  # Prediction:
  xgb.pred <- predict(xgb_model, xtest)
  traipre= predict(xgb_model,xtrain)
  
  #' Confusion Matrix
  
  xgb.pred <- as.factor(xgb.pred)
  ytest <- as.factor(ytest)
  
  # Ensure levels match
  levels(xgb.pred) <- levels(ytest)
  
  # Compute confusion matrix
  y=confusionMatrix(xgb.pred, ytest)
  y1=confusionMatrix(as.factor(traipre),as.factor(ytrain))
  testac=as.numeric(y$overall['Accuracy'])
  trainac=as.numeric(y1$overall['Accuracy'])
  
  gtest[i]=testac
  gtrain[i]=trainac
  
  if (testac >=0.95 ) { 
    be[[i]]=xgb_tune$bestTune
    g[i]=abs(testac-trainac)
    
  }else{
    be[[i]]=NULL
    g[i]=Inf
  }  
}  
z=which.max(gtest)
best=be[z]

# Train control (no resampling now)
final_train_control <- trainControl(method = "none", verboseIter = TRUE)

# Final training with best hyperparameters
xgb_best_model <- train(x = xtrain,
                        y = ytrain,
                        trControl = final_train_control,
                        tuneGrid = best[[1]],  # be[z] is a list, so extract with [[1]]
                        method = "xgbTree",
                        verbose = TRUE)

# Predict on test and training sets
final_pred_test <- predict(xgb_best_model, xtest)
final_pred_train <- predict(xgb_best_model, xtrain)

# Confusion matrices
cat("Confusion Matrix - Test:\n")
print(confusionMatrix(final_pred_test, ytest))

cat("\nConfusion Matrix - Train:\n")
print(confusionMatrix(final_pred_train, ytrain))


compute_f1 <- function(cm) {
  precision <- cm$byClass[, "Pos Pred Value"]   # precision per class
  recall <- cm$byClass[, "Sensitivity"]        # recall per class
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(f1)
}

# For test
cm_test <- confusionMatrix(final_pred_test, ytest)
f1_test <- compute_f1(cm_test)
print(f1_test)

# For train
cm_train <- confusionMatrix(final_pred_train, ytrain)
f1_train <- compute_f1(cm_train)
print(f1_train)

library(xgboost)
importance_matrix <- xgb.importance(model = xgb_best_model$finalModel)
print(importance_matrix)

# Plot importance
xgb.plot.importance(importance_matrix)


library(ggplot2)

importance_matrix <- xgb.importance(model = xgb_best_model$finalModel)

ggplot(importance_matrix, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#1f78b4", color = "black", alpha = 0.8) +    # nice blue bars
  coord_flip() +
  labs(title = "XGBoost Feature Importance",
       subtitle = "Gain metric",
       x = "Feature",
       y = "Importance (Gain)") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.title = element_text(face = "bold"))

library(pdp)
library(ggplot2)

# Variables to plot (numeric columns)
vars_to_plot <- numeric_cols  # from your earlier code

# Create empty list to store plots
pdp_plots <- list()

# Loop over variables
for (var in vars_to_plot) {
  
  # Compute PDP for this variable
  pdp_var <- partial(xgb_best_model, 
                     pred.var = var, 
                     which.class = "Hazardous", 
                     prob = TRUE,
                     train = xtrain)
  
  # Convert back to original scale
  mean_val <- train_mean[var]
  sd_val <- train_sd[var]
  pdp_var[[paste0(var, "_original")]] <- pdp_var[[var]] * sd_val + mean_val
  
  # Plot with ggplot2 using original scale on x-axis
  p <- ggplot(pdp_var, aes_string(x = paste0(var, "_original"), y = "yhat")) +
    geom_line(color = "#e41a1c", size = 1.3) +
    geom_point(color = "#e41a1c", size = 2) +
    labs(title = paste0("PDP for ", var, " (Hazardous Class)"),
         x = var,
         y = "Predicted Probability") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          axis.title = element_text(face = "bold"))
  
 
  pdp_plots[[var]] <- p
  
  print(p)
}

