## Air Quality Level Prediction

**Dataset:** Air Quality Dataset (Kaggle, 5,000 observations)  
**Tools:** R (ggplot2, dplyr, FactoMineR, xgboost, randomForest, e1071)

### Objective
Classify regional air quality into four levels — Good, Moderate, Poor, Hazardous — 
based on pollutant concentrations and environmental factors.

### Methods
- Exploratory Data Analysis (EDA)
- Phi-K Correlation Analysis
- VIF-based Multicollinearity Analysis
- Principal Component Analysis (PCA) — 9 variables reduced to 7 components retaining 97% variance
- Outlier Detection: Isolation Forest + Robust Mahalanobis Distance
- SMOTE to address class imbalance
- Models: XGBoost, Random Forest, SVM, Neural Network, Logistic Regression

### Key Results
- XGBoost achieved best performance: 96% test accuracy, Hazardous F1-score: 0.88
- PM2.5 and PM10 showed near-perfect multicollinearity (r = 0.99, VIF > 25) — resolved via PCA
- Top predictors: CO, proximity to industrial areas, NO2, SO2
- Partial Dependence Plots used to interpret model predictions
