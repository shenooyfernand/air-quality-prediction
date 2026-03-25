# ============================================
# 2. PREPROCESSING
# ============================================

library(caret)
library(smotefamily)

data <- data[-1]  # remove ID

# Encode target
label_map <- c("Good"=1, "Moderate"=2, "Poor"=3, "Hazardous"=4)
data$Air.Quality <- as.numeric(label_map[data$Air.Quality])

# Train/test split
set.seed(123)
train_index <- sample(1:nrow(data), 0.8*nrow(data))

train <- data[train_index, ]
test  <- data[-train_index, ]

# Scale
num_cols <- setdiff(names(train), "Air.Quality")

mean_vals <- sapply(train[, num_cols], mean)
sd_vals   <- sapply(train[, num_cols], sd)

train_scaled <- scale(train[, num_cols], center=mean_vals, scale=sd_vals)
test_scaled  <- scale(test[, num_cols], center=mean_vals, scale=sd_vals)

train_df <- as.data.frame(train_scaled)
train_df$Air.Quality <- train$Air.Quality

test_df <- as.data.frame(test_scaled)
test_df$Air.Quality <- test$Air.Quality

# SMOTE
smote_data <- SMOTE(train_df[, num_cols], train_df$Air.Quality)

train_bal <- smote_data$data
colnames(train_bal)[ncol(train_bal)] <- "Air.Quality"

# Convert back to factor
inv_map <- c("1"="Good","2"="Moderate","3"="Poor","4"="Hazardous")

train_bal$Air.Quality <- factor(inv_map[as.character(train_bal$Air.Quality)])

xtrain <- train_bal[, num_cols]
ytrain <- train_bal$Air.Quality

xtest <- test_df[, num_cols]
ytest <- factor(inv_map[as.character(test_df$Air.Quality)])