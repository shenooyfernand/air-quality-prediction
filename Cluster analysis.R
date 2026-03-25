# ============================================
# CLUSTER ANALYSIS
# ============================================

library(cluster)
library(factoextra)

# Use scaled data
cluster_data <- scale(data[, num_cols])

# ----------------------------
# Optimal clusters
# ----------------------------

fviz_nbclust(cluster_data, kmeans, method="silhouette")

# ----------------------------
# K-means
# ----------------------------

set.seed(123)
kmeans_model <- kmeans(cluster_data, centers=2, nstart=25)

# Visualization
fviz_cluster(kmeans_model, data=cluster_data)

# ----------------------------
# Interpretation
# ----------------------------

data$cluster <- as.factor(kmeans_model$cluster)

aggregate(. ~ cluster, data=data, mean)