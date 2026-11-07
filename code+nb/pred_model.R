# ==============================================================================
# Coffee Quality Prediction Pipeline - Comprehensive Prediction Script
# Includes Step-by-Step Summary Tables and Visualizations
# ==============================================================================

# Load necessary libraries
library(dplyr)
library(caret)
library(leaps)
library(FNN)          # Used for KNN regression
library(randomForest) # Used for Random Forest
library(readr)

# Step 1: Setup, Data Loading, and Preprocessing

coffee_ratings <- read_csv(here::here("data_info", "coffee_ratings.csv"))

# Select required columns
cols_to_use <- c("total_cup_points", "moisture", "category_one_defects", 
                 "category_two_defects", "altitude_mean_meters", "color", 
                 "country_of_origin", "species", "variety", "processing_method")

df <- coffee_ratings[, cols_to_use]

# Impute missing values for categorical variables with "unknown"
cat_vars <- c("color", "country_of_origin", "species", "variety", "processing_method")
for (var in cat_vars) {
  df[[var]] <- as.character(df[[var]])
  df[[var]][is.na(df[[var]]) | df[[var]] == "" | df[[var]] == "NA"] <- "unknown"
  df[[var]] <- as.factor(df[[var]])
}

# Group low-frequency categories (< 5 lots) into "other"
for (var in cat_vars) {
  counts <- table(df[[var]])
  rare_cats <- names(counts[counts < 5])
  levels_df <- levels(df[[var]])
  levels_df[levels_df %in% rare_cats] <- "other"
  levels(df[[var]]) <- levels_df
}

# Make sure the numeric variables are actually numeric
numeric_vars <- c("total_cup_points", "moisture", "category_one_defects", 
                  "category_two_defects", "altitude_mean_meters")
for (var in numeric_vars) {
  df[[var]] <- as.numeric(df[[var]])
}

# Remove remaining rows with missing values in numeric variables
df <- na.omit(df)

cat("=== STEP 1 COMPLETE: Preprocessed Dataset Dimensions ==\n")
print(dim(df))

# VISUALIZATION 1: Distribution of the Target Variable (Total Cup Points)
hist(df$total_cup_points, main = "Distribution of Coffee Cupping Scores",
     xlab = "Total Cup Points", col = "skyblue", border = "white", breaks = 20)

# Step 2: Data Splitting (60% Train, 20% Validation, 20% Test)

set.seed(42) # For reproducibility
n <- nrow(df)

train_index <- sample(1:n, size = 0.60 * n)
remaining_index <- setdiff(1:n, train_index)
val_index <- sample(remaining_index, size = 0.50 * length(remaining_index)) # 20% of total
test_index <- setdiff(remaining_index, val_index)                           # 20% of total

train_data <- df[train_index, ]
val_data   <- df[val_index, ]
test_data  <- df[test_index, ]

# SUMMARY TABLE 1: Split Distribution & Target Statistics
split_summary_table <- data.frame(
  Subset = c("Training", "Validation", "Test"),
  Count = c(nrow(train_data), nrow(val_data), nrow(test_data)),
  Percentage = c(nrow(train_data)/n*100, nrow(val_data)/n*100, nrow(test_data)/n*100),
  Mean_Score = c(mean(train_data$total_cup_points), mean(val_data$total_cup_points), mean(test_data$total_cup_points)),
  SD_Score = c(sd(train_data$total_cup_points), sd(val_data$total_cup_points), sd(test_data$total_cup_points))
)

cat("\n=== SUMMARY TABLE 1: Dataset Partitioning & Target Statistics ===\n")
print(split_summary_table)

# Step 3: Z-Score Standardization for Numeric Predictors

num_preds <- c("moisture", "category_one_defects", "category_two_defects", "altitude_mean_meters")

# Compute means and standard deviations exclusively from training data
train_means <- sapply(train_data[, num_preds], mean)
train_sds   <- sapply(train_data[, num_preds], sd)

standardize_cols <- function(data, preds, means, sds) {
  scaled_df <- data
  for (p in preds) {
    scaled_df[[p]] <- (scaled_df[[p]] - means[p]) / sds[p]
  }
  return(scaled_df)
}

train_data_scaled <- standardize_cols(train_data, num_preds, train_means, train_sds)
val_data_scaled   <- standardize_cols(val_data, num_preds, train_means, train_sds)
test_data_scaled  <- standardize_cols(test_data, num_preds, train_means, train_sds)

# Step 4: Model Training, Tuning, and Intermediate Visualizations

# Baseline Model
baseline_pred <- mean(train_data$total_cup_points)
baseline_rmse <- sqrt(mean((test_data$total_cup_points - baseline_pred)^2))

# Multiple Linear Regression
lm_model <- lm(total_cup_points ~ ., data = train_data_scaled)
lm_preds <- predict(lm_model, newdata = val_data_scaled)
lm_val_rmse <- sqrt(mean((val_data_scaled$total_cup_points - lm_preds)^2))

# Forward Stepwise Selection
max_vars <- min(15, ncol(model.matrix(total_cup_points ~ ., data = train_data_scaled)) - 1)
regfit.full <- regsubsets(total_cup_points ~ ., data = train_data_scaled, 
                          nvmax = max_vars, method = "forward")
reg_summary <- summary(regfit.full)

num_models_fit <- length(reg_summary$cp)
val_mses_step <- numeric(num_models_fit)

for (i in 1:num_models_fit) {
  coefs <- coef(regfit.full, id = i)
  common_vars <- intersect(names(coefs), colnames(model.matrix(total_cup_points ~ ., data = val_data_scaled)))
  X_val_subset <- model.matrix(total_cup_points ~ ., data = val_data_scaled)[, common_vars, drop = FALSE]
  preds <- X_val_subset %*% coefs[common_vars]
  val_mses_step[i] <- mean((val_data_scaled$total_cup_points - preds)^2)
}

best_k <- which.min(val_mses_step)

# VISUALIZATION 2: Stepwise Validation MSE Curve
plot(1:num_models_fit, val_mses_step, type = "b", pch = 19, col = "darkblue",
     xlab = "Number of Predictors (k)", ylab = "Validation MSE",
     main = "Forward Stepwise Selection: Validation MSE vs. Model Size")
points(best_k, val_mses_step[best_k], col = "red", pch = 19, cex = 1.5)
legend("topright", legend = c("Validation MSE", paste("Optimal k =", best_k)), 
       col = c("darkblue", "red"), pch = c(19, 19))

# KNN Regression (Tuning k from 1 to 50)
train_matrix <- model.matrix(total_cup_points ~ . - 1, data = train_data_scaled)
val_matrix   <- model.matrix(total_cup_points ~ . - 1, data = val_data_scaled)
test_matrix  <- model.matrix(total_cup_points ~ . - 1, data = test_data_scaled)

k_values <- 1:50
val_rmse_knn <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  knn_pred <- knn.reg(train = train_matrix, test = val_matrix, 
                      y = train_data_scaled$total_cup_points, k = k_values[i])$pred
  val_rmse_knn[i] <- sqrt(mean((val_data_scaled$total_cup_points - knn_pred)^2))
}

best_k_knn <- k_values[which.min(val_rmse_knn)]

# VISUALIZATION 3: KNN Tuning Curve (RMSE vs K)
plot(k_values, val_rmse_knn, type = "l", lwd = 2, col = "forestgreen",
     xlab = "Number of Neighbors (k)", ylab = "Validation RMSE",
     main = "KNN Regression Tuning: Validation RMSE across K")
points(best_k_knn, min(val_rmse_knn), col = "red", pch = 19, cex = 1.5)
legend("topright", legend = c("Validation RMSE", paste("Optimal k =", best_k_knn)), 
       col = c("forestgreen", "red"), lty = c(1, NA), pch = c(NA, 19))

# Random Forest (Tuning mtry)
set.seed(42)
mtry_values <- c(2, 4, 6, 8)
val_rmse_rf <- numeric(length(mtry_values))

for (i in seq_along(mtry_values)) {
  rf_model <- randomForest(total_cup_points ~ ., data = train_data_scaled, 
                           ntree = 300, mtry = mtry_values[i], bootstrap = TRUE)
  rf_preds <- predict(rf_model, newdata = val_data_scaled)
  val_rmse_rf[i] <- sqrt(mean((val_data_scaled$total_cup_points - rf_preds)^2))
}

best_mtry <- mtry_values[which.min(val_rmse_rf)]

# VISUALIZATION 4: Random Forest Tuning (RMSE vs mtry)
plot(mtry_values, val_rmse_rf, type = "b", pch = 19, col = "purple",
     xlab = "Parameter mtry", ylab = "Validation RMSE",
     main = "Random Forest Tuning: Validation RMSE vs. mtry")
points(best_mtry, val_rmse_rf[which.min(val_rmse_rf)], col = "red", pch = 19, cex = 1.5)

# Fit Final Random Forest Model with optimal mtry
final_rf <- randomForest(total_cup_points ~ ., data = train_data_scaled, 
                         ntree = 500, mtry = best_mtry, bootstrap = TRUE)

# VISUALIZATION 5: Random Forest Variable Importance Plot
varImpPlot(final_rf, main = "Random Forest Variable Importance")

# ------------------------------------------------------------------------------
# Step 5: Final Evaluation on the Test Set & Summary Reporting
# ------------------------------------------------------------------------------

evaluate_model <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r2 <- 1 - (ss_res / ss_tot)
  return(c(RMSE = rmse, R2 = r2))
}

# Baseline Test Performance
baseline_test_preds <- rep(mean(train_data$total_cup_points), nrow(test_data))
res_baseline <- evaluate_model(test_data$total_cup_points, baseline_test_preds)

# MLR Test Performance
lm_test_preds <- predict(lm_model, newdata = test_data_scaled)
res_lm <- evaluate_model(test_data$total_cup_points, lm_test_preds)

# Stepwise Test Performance
step_coefs <- coef(regfit.full, id = best_k)
test_mat_step <- model.matrix(total_cup_points ~ ., data = test_data_scaled)[, names(step_coefs)]
step_test_preds <- test_mat_step %*% step_coefs
res_step <- evaluate_model(test_data$total_cup_points, step_test_preds)

# KNN Test Performance
knn_test_preds <- knn.reg(train = train_matrix, test = test_matrix, 
                          y = train_data_scaled$total_cup_points, k = best_k_knn)$pred
res_knn <- evaluate_model(test_data$total_cup_points, knn_test_preds)

# Random Forest Test Performance
rf_test_preds <- predict(final_rf, newdata = test_data_scaled)
res_rf <- evaluate_model(test_data$total_cup_points, rf_test_preds)

# SUMMARY TABLE 2: Final Model Performance Comparison Table
results_table <- data.frame(
  Model = c("Baseline", "Multiple Linear Regression", "Stepwise Regression", "KNN Regression", "Random Forest"),
  Test_RMSE = c(res_baseline["RMSE"], res_lm["RMSE"], res_step["RMSE"], res_knn["RMSE"], res_rf["RMSE"]),
  Test_R2 = c(res_baseline["R2"], res_lm["R2"], res_step["R2"], res_knn["R2"], res_rf["R2"])
)

cat("\n=== SUMMARY TABLE 2: Final Test Set Performance Comparison ===\n")
print(results_table)

# VISUALIZATION 6: Final Performance Comparison Bar Chart
par(mar = c(6, 4, 4, 2))
barplot(results_table$Test_RMSE, names.arg = results_table$Model, las = 2, col = "coral",
        ylab = "Test RMSE (Lower is Better)", main = "Comparison of Test RMSE Across Models",
        cex.names = 0.8)
abline(h = res_baseline["RMSE"], lty = 2, col = "black")
legend("topright", legend = c("Baseline RMSE Threshold"), lty = 2, col = "black", cex = 0.8)
par(mar = c(5, 4, 4, 2)) # Reset default margins