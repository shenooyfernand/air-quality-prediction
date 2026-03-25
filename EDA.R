# ============================================
# 1. EDA
# ============================================

library(ggplot2)
library(dplyr)
library(corrplot)

data <- read.csv("train.csv")

# Target distribution
ggplot(data, aes(x = Air.Quality)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Distribution of Air Quality")

# Summary
summary(data)

# ----------------------------
# Univariate Analysis
# ----------------------------

num_cols <- sapply(data, is.numeric)

for (col in names(data)[num_cols]) {
  print(
    ggplot(data, aes_string(x = col)) +
      geom_histogram(fill="orange", bins=30) +
      ggtitle(paste("Distribution of", col))
  )
}

# ----------------------------
# Bivariate Analysis
# ----------------------------

# Boxplots vs target
for (col in names(data)[num_cols]) {
  print(
    ggplot(data, aes_string(x="Air.Quality", y=col)) +
      geom_boxplot() +
      ggtitle(paste(col, "vs Air Quality"))
  )
}

# ----------------------------
# Correlation
# ----------------------------

cor_mat <- cor(data[, num_cols])
corrplot(cor_mat, method="color")

# ----------------------------
# Outlier detection (basic)
# ----------------------------

boxplot(data$Humidity, main="Humidity Outliers")

# Remove invalid humidity (>100)
data <- data[data$Humidity <= 100, ]