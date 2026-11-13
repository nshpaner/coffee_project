# ==============================================================================
# Coffee Quality Prediction Pipeline Script

# Research Question: Can we learn a function Y = f(X) to accurately predict a 
# coffee lot’s overall quality score (total_cup_points) using only its non-sensory  
# physical, agronomic, and geographic attributes (X)?
# ==============================================================================

library(dplyr)
library(leaps)
library(FNN)          # KNN regression
library(randomForest) # Random Forest
library(readr)

# Step 1: Data Loading & Cleaning
# ==============================================================================
# This first step loads the raw coffee ratings dataset, extracts all of our   
# specified target variable and predictor columns, converts character  
# variables to factors, and cleans numeric formats.

# This makes sure that the data types are correctly formatted and ready for  
# modeling and filters out any missing data records to prevent bugs
# and errors down the line.
# ==============================================================================

coffee_ratings <- read_csv(here::here("data_info", "coffee_ratings.csv"), show_col_types = FALSE)

cols_to_use <- c("total_cup_points", "moisture", "category_one_defects", 
                 "category_two_defects", "altitude_mean_meters", "color", 
                 "country_of_origin", "species", "variety", "processing_method")

df <- coffee_ratings[, cols_to_use]

cat_vars <- c("color", "country_of_origin", "species", "variety", "processing_method")
for (var in cat_vars) {
  df[[var]] <- as.character(df[[var]])
  df[[var]][is.na(df[[var]]) | df[[var]] == "" | df[[var]] == "NA"] <- "unknown"
}

numeric_vars <- c("total_cup_points", "moisture", "category_one_defects", 
                  "category_two_defects", "altitude_mean_meters")
for (var in numeric_vars) {
  df[[var]] <- as.numeric(df[[var]])
}

df <- na.omit(df)

# Filter out all rows with 0 cup points
df <- df[df$total_cup_points > 0, ]

cat("=== STEP 1 COMPLETE: Cleaned Dataset Dimensions ===\n")
print(dim(df))

# Step 2: 60/20/20 Train / Validation / Test Split (Before Feature Eng.)
# ==============================================================================
# Now, we will randomly partition the dataset into a 60% training set, 
# 20% validation set, and 20% held-out test set using a fixed random seed.

# This is quite important for both preventing optimism bias (hence the CV approach) 
# as well as data leakage from not saving random results. The training set is 
# used for fitting, the validation set is used only for hyperparameter 
# tuning and model selection, and the test set will be completely unseen 
# until the final unbiased evaluation.
# ==============================================================================

set.seed(42)
n <- nrow(df)

train_index     <- sample(1:n, size = floor(0.60 * n))
remaining_index <- setdiff(1:n, train_index)
val_index       <- sample(remaining_index, size = floor(0.50 * length(remaining_index)))
test_index      <- setdiff(remaining_index, val_index)

train_data <- df[train_index, ]
val_data   <- df[val_index, ]
test_data  <- df[test_index, ]

split_summary_table <- data.frame(
  Subset     = c("Training", "Validation", "Test"),
  Count      = c(nrow(train_data), nrow(val_data), nrow(test_data)),
  Percentage = round(c(nrow(train_data), nrow(val_data), nrow(test_data)) / n * 100, 1),
  Mean_Score = round(c(mean(train_data$total_cup_points), mean(val_data$total_cup_points), mean(test_data$total_cup_points)), 3),
  SD_Score   = round(c(sd(train_data$total_cup_points), sd(val_data$total_cup_points), sd(test_data$total_cup_points)), 3)
)

cat("\n=== SUMMARY TABLE 1: Dataset Partitioning & Target Statistics ===\n")
print(split_summary_table)

# Step 3: Feature Engineering
# ==============================================================================
# This leg of the script learns categorical factor levels of a substantial enough 
# size (frequencies >= 5), as well as means and standard deviations ONLY
# from the training set. It then maps them onto validation/test splits
# and standardizes the numeric features.

# In doing so, we prevent data from meshing from validation/test sets into the
# training set, and avoids "new factor level" errors by collapsing
# all of the rare categories into "other".
# ==============================================================================

for (var in cat_vars) {
  train_counts <- table(train_data[[var]])
  valid_levels <- names(train_counts[train_counts >= 5])
  
  map_rare <- function(x) {
    x_clean <- ifelse(x %in% valid_levels, x, "other")
    factor(x_clean, levels = unique(c(valid_levels, "other")))
  }
  
  train_data[[var]] <- map_rare(train_data[[var]])
  val_data[[var]]   <- map_rare(val_data[[var]])
  test_data[[var]]  <- map_rare(test_data[[var]])
}

# Standardize numeric predictors using only the training means and SDs
num_preds   <- c("moisture", "category_one_defects", "category_two_defects", "altitude_mean_meters")
train_means <- sapply(train_data[, num_preds], mean)
train_sds   <- sapply(train_data[, num_preds], sd)

standardize_cols <- function(data, preds, means, sds) {
  scaled_df <- data
  for (p in preds) {
    scaled_df[[p]] <- (scaled_df[[p]] - means[p]) / sds[p]
  }
  return(scaled_df)
}

train_scaled <- standardize_cols(train_data, num_preds, train_means, train_sds)
val_scaled   <- standardize_cols(val_data, num_preds, train_means, train_sds)
test_scaled  <- standardize_cols(test_data, num_preds, train_means, train_sds)

# This is a helper function for safe subset matrix prediction
predict_regsubsets <- function(regfit_obj, id, newdata) {
  coefs <- coef(regfit_obj, id = id)
  X_mat <- model.matrix(total_cup_points ~ ., data = newdata)
  common_vars <- intersect(names(coefs), colnames(X_mat))
  as.vector(X_mat[, common_vars, drop = FALSE] %*% coefs[common_vars])
}

# Step 4: Model Training, Tuning & Bias-Variance (Train vs. Validation) Curves
# ==============================================================================
# This is where the fun begins! Next, we fit the baseline, stepwise, KNN, and   
# random forest models, tune parameters (subset size k, neighbors k, mtry,   
# tree depth maxnodes) while tracking both the Training and Validation RMSE.

# The process directly measures and plots the bias-variance tradeoff.  
# Visualizing training vs. validation error curves allows us to see where  
# models begin overfitting to noise.
# We can then find the optimal parameters to use for each model.
# ==============================================================================

# OLS Multiple Linear Regression
lm_model    <- lm(total_cup_points ~ ., data = train_scaled)
lm_train_rmse <- sqrt(mean((train_scaled$total_cup_points - predict(lm_model, train_scaled))^2))
lm_val_rmse   <- sqrt(mean((val_scaled$total_cup_points - predict(lm_model, val_scaled))^2))

# Forward Stepwise Selection (Train vs. Validation RMSE across model size k)
max_vars    <- min(20, ncol(model.matrix(total_cup_points ~ ., data = train_scaled)) - 1)
regfit_fwd  <- regsubsets(total_cup_points ~ ., data = train_scaled, nvmax = max_vars, method = "forward")
num_models  <- length(summary(regfit_fwd)$cp)

step_train_rmse <- numeric(num_models)
step_val_rmse   <- numeric(num_models)

for (i in 1:num_models) {
  step_train_rmse[i] <- sqrt(mean((train_scaled$total_cup_points - predict_regsubsets(regfit_fwd, i, train_scaled))^2))
  step_val_rmse[i]   <- sqrt(mean((val_scaled$total_cup_points - predict_regsubsets(regfit_fwd, i, val_scaled))^2))
}
best_k_step <- which.min(step_val_rmse)

# Graph 1: Stepwise Train vs. Validation RMSE
plot(1:num_models, step_val_rmse, type = "b", pch = 19, col = "darkblue",
     ylim = range(c(step_train_rmse, step_val_rmse)),
     xlab = "Number of Predictors (Model Complexity)", ylab = "RMSE",
     main = "Forward Stepwise: Training vs. Validation RMSE")
lines(1:num_models, step_train_rmse, type = "b", pch = 17, lty = 2, col = "darkgray")
points(best_k_step, step_val_rmse[best_k_step], col = "red", pch = 19, cex = 1.5)
legend("topright", legend = c("Validation RMSE", "Training RMSE", paste("Optimal Size =", best_k_step)),
       col = c("darkblue", "darkgray", "red"), lty = c(1, 2, NA), pch = c(19, 17, 19), cex = 0.85)

# KNN Regression (Train vs. Validation RMSE across k = 1..50)
train_matrix <- model.matrix(total_cup_points ~ . - 1, data = train_scaled)
val_matrix   <- model.matrix(total_cup_points ~ . - 1, data = val_scaled)
test_matrix  <- model.matrix(total_cup_points ~ . - 1, data = test_scaled)

k_values       <- 1:50
knn_train_rmse <- numeric(length(k_values))
knn_val_rmse   <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  tr_pred <- knn.reg(train = train_matrix, test = train_matrix, y = train_scaled$total_cup_points, k = k_values[i])$pred
  va_pred <- knn.reg(train = train_matrix, test = val_matrix,   y = train_scaled$total_cup_points, k = k_values[i])$pred
  knn_train_rmse[i] <- sqrt(mean((train_scaled$total_cup_points - tr_pred)^2))
  knn_val_rmse[i]   <- sqrt(mean((val_scaled$total_cup_points - va_pred)^2))
}
best_k_knn <- k_values[which.min(knn_val_rmse)]

# Graph 2: KNN Train vs. Validation RMSE (Bias-Variance Tradeoff)
plot(k_values, knn_val_rmse, type = "l", lwd = 2, col = "forestgreen",
     ylim = range(c(knn_train_rmse, knn_val_rmse)),
     xlab = "Number of Neighbors (k) [High Flexibility -> Low Flexibility]", ylab = "RMSE",
     main = "KNN Regression: Training vs. Validation RMSE")
lines(k_values, knn_train_rmse, lwd = 2, lty = 2, col = "darkgray")
points(best_k_knn, min(knn_val_rmse), col = "red", pch = 19, cex = 1.5)
legend("bottomright", legend = c("Validation RMSE", "Training RMSE", paste("Optimal k =", best_k_knn)),
       col = c("forestgreen", "darkgray", "red"), lty = c(1, 2, NA), pch = c(NA, NA, 19), cex = 0.85)

# Random Forest Tuning (mtry AND Tree Depth; maxnodes for different parts of the bias/variance curve)
set.seed(42)
mtry_values   <- c(2, 4, 6, 8)
rf_train_rmse <- numeric(length(mtry_values))
rf_val_rmse   <- numeric(length(mtry_values))

for (i in seq_along(mtry_values)) {
  rf_tmp <- randomForest(total_cup_points ~ ., data = train_scaled, ntree = 300, mtry = mtry_values[i])
  rf_train_rmse[i] <- sqrt(mean((train_scaled$total_cup_points - predict(rf_tmp, train_scaled))^2))
  rf_val_rmse[i]   <- sqrt(mean((val_scaled$total_cup_points - predict(rf_tmp, val_scaled))^2))
}
best_mtry <- mtry_values[which.min(rf_val_rmse)]

# Flexibility Curve at best_mtry (the maxnodes)
maxnodes_grid    <- c(5, 10, 20, 40, 80, 150, 250)
depth_train_rmse <- numeric(length(maxnodes_grid))
depth_val_rmse   <- numeric(length(maxnodes_grid))

for (i in seq_along(maxnodes_grid)) {
  set.seed(42)
  rf_d <- randomForest(total_cup_points ~ ., data = train_scaled, ntree = 300, 
                       mtry = best_mtry, maxnodes = maxnodes_grid[i])
  depth_train_rmse[i] <- sqrt(mean((train_scaled$total_cup_points - predict(rf_d, train_scaled))^2))
  depth_val_rmse[i]   <- sqrt(mean((val_scaled$total_cup_points - predict(rf_d, val_scaled))^2))
}
best_maxnodes <- maxnodes_grid[which.min(depth_val_rmse)]

# Graph 3: Random Forest Flexibility Curve
plot(maxnodes_grid, depth_val_rmse, type = "b", pch = 19, col = "purple",
     ylim = range(c(depth_train_rmse, depth_val_rmse)),
     xlab = "Max Terminal Nodes (Tree Depth / Flexibility)", ylab = "RMSE",
     main = "Random Forest: Training vs. Validation RMSE across Tree Depth")
lines(maxnodes_grid, depth_train_rmse, type = "b", pch = 17, lty = 2, col = "darkgray")
points(best_maxnodes, min(depth_val_rmse), col = "red", pch = 19, cex = 1.5)
legend("topright", legend = c("Validation RMSE", "Training RMSE", paste("Optimal maxnodes =", best_maxnodes)),
       col = c("purple", "darkgray", "red"), lty = c(1, 2, NA), pch = c(19, 17, 19), cex = 0.85)

# The final tuned Random Forest model
set.seed(42)
final_rf <- randomForest(total_cup_points ~ ., data = train_scaled, ntree = 500, 
                         mtry = best_mtry, maxnodes = best_maxnodes, importance = TRUE)

# Graph 4: Random Forest Variable Importance Plot (VIP)
varImpPlot(final_rf, main = "Tuned Random Forest Variable Importance", cex = 0.8)

# Step 5: Final Evaluation on the Test Set: Analyzing Overfitting
# ==============================================================================
# The final part of this script pertains to the model evaluation. Here, we
# analyze all finalized models on the test set, calculating Test RMSE, Test R^2, 
# and what we call the "Overfit Gap" (which refers to Test - Train RMSE).

# This completes our prediction analysis by quantifying how well our models perform 
# on unseen data, and proving that hyperparameter tuning successfully controlled 
# for optimism bias.
# ==============================================================================

evaluate_model <- function(actual, predicted) {
  rmse   <- sqrt(mean((actual - predicted)^2))
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r2     <- 1 - (ss_res / ss_tot)
  return(c(RMSE = rmse, R2 = r2))
}

# Predictions across Train and Test sets
base_pred_tr <- rep(mean(train_scaled$total_cup_points), nrow(train_scaled))
base_pred_te <- rep(mean(train_scaled$total_cup_points), nrow(test_scaled))

lm_pred_tr   <- predict(lm_model, newdata = train_scaled)
lm_pred_te   <- predict(lm_model, newdata = test_scaled)

step_pred_tr <- predict_regsubsets(regfit_fwd, best_k_step, train_scaled)
step_pred_te <- predict_regsubsets(regfit_fwd, best_k_step, test_scaled)

knn_pred_tr  <- knn.reg(train = train_matrix, test = train_matrix, y = train_scaled$total_cup_points, k = best_k_knn)$pred
knn_pred_te  <- knn.reg(train = train_matrix, test = test_matrix,  y = train_scaled$total_cup_points, k = best_k_knn)$pred

rf_pred_tr   <- predict(final_rf, newdata = train_scaled)
rf_pred_te   <- predict(final_rf, newdata = test_scaled)

res_base <- evaluate_model(test_scaled$total_cup_points, base_pred_te)
res_lm   <- evaluate_model(test_scaled$total_cup_points, lm_pred_te)
res_step <- evaluate_model(test_scaled$total_cup_points, step_pred_te)
res_knn  <- evaluate_model(test_scaled$total_cup_points, knn_pred_te)
res_rf   <- evaluate_model(test_scaled$total_cup_points, rf_pred_te)

train_rmses_final <- c(
  sqrt(mean((train_scaled$total_cup_points - base_pred_tr)^2)),
  sqrt(mean((train_scaled$total_cup_points - lm_pred_tr)^2)),
  sqrt(mean((train_scaled$total_cup_points - step_pred_tr)^2)),
  sqrt(mean((train_scaled$total_cup_points - knn_pred_tr)^2)),
  sqrt(mean((train_scaled$total_cup_points - rf_pred_tr)^2))
)

results_table <- data.frame(
  Model         = c("Baseline (Mean)", "Multiple Linear Regression", "Forward Stepwise", "KNN Regression", "Tuned Random Forest"),
  Train_RMSE    = round(train_rmses_final, 4),
  Test_RMSE     = round(c(res_base["RMSE"], res_lm["RMSE"], res_step["RMSE"], res_knn["RMSE"], res_rf["RMSE"]), 4),
  Overfit_Gap   = round(c(res_base["RMSE"], res_lm["RMSE"], res_step["RMSE"], res_knn["RMSE"], res_rf["RMSE"]) - train_rmses_final, 4),
  Test_R2       = round(c(res_base["R2"], res_lm["R2"], res_step["R2"], res_knn["R2"], res_rf["R2"]), 4)
)

cat("\n=== SUMMARY TABLE 2: Final Train vs. Test Performance & Overfit Gap ===\n")
print(results_table)

# Graph 5: Final Test RMSE Comparison Bar Chart
par(mar = c(10, 4, 4, 2)) # Increased bottom margin from 7 to 10 for long labels

barplot(results_table$Test_RMSE, 
        names.arg = c("Baseline (Mean)", "Multiple Linear Reg.", "Forward Stepwise", "KNN Regression", "Tuned Random Forest"), 
        las = 2, col = "coral",
        ylab = "Test RMSE (Lower is Better)", 
        main = "Comparison of Test RMSE Across Models",
        cex.names = 0.75,
        ylim = c(0, max(results_table$Test_RMSE) * 1.15))

abline(h = res_base["RMSE"], lty = 2, col = "black", lwd = 1.5)

legend("topright", legend = c("Baseline Benchmark RMSE"), lty = 2, col = "black", lwd = 1.5, cex = 0.8)

par(mar = c(5, 4, 4, 2))

# ==============================================================================
# EXPORT & REPORTING: Save Prediction Results & Visualizations to pred_results/
# ==============================================================================

# Create output directory if it doesn't exist
if (!dir.exists("pred_results")) {
  dir.create("pred_results")
}

# Save Summary Tables as CSV files
write.csv(split_summary_table, "pred_results/split_summary_table.csv", row.names = FALSE)
write.csv(results_table, "pred_results/final_performance_results.csv", row.names = FALSE)

# Re-render and save all key figures to pred_results/ for markdown inclusion
pdf("pred_results/stepwise_train_val_rmse.pdf", width = 9, height = 6)
plot(1:num_models, step_val_rmse, type = "b", pch = 19, col = "darkblue",
     ylim = range(c(step_train_rmse, step_val_rmse)),
     xlab = "Number of Predictors (Model Complexity)", ylab = "RMSE",
     main = "Forward Stepwise: Training vs. Validation RMSE")
lines(1:num_models, step_train_rmse, type = "b", pch = 17, lty = 2, col = "darkgray")
points(best_k_step, step_val_rmse[best_k_step], col = "red", pch = 19, cex = 1.5)
legend("topright", legend = c("Validation RMSE", "Training RMSE", paste("Optimal Size =", best_k_step)),
       col = c("darkblue", "darkgray", "red"), lty = c(1, 2, NA), pch = c(19, 17, 19), cex = 0.85)
dev.off()

pdf("pred_results/knn_train_val_rmse.pdf", width = 9, height = 6)
plot(k_values, knn_val_rmse, type = "l", lwd = 2, col = "forestgreen",
     ylim = range(c(knn_train_rmse, knn_val_rmse)),
     xlab = "Number of Neighbors (k)", ylab = "RMSE",
     main = "KNN Regression: Training vs. Validation RMSE")
lines(k_values, knn_train_rmse, lwd = 2, lty = 2, col = "darkgray")
points(best_k_knn, min(knn_val_rmse), col = "red", pch = 19, cex = 1.5)
legend("bottomright", legend = c("Validation RMSE", "Training RMSE", paste("Optimal k =", best_k_knn)),
       col = c("forestgreen", "darkgray", "red"), lty = c(1, 2, NA), pch = c(NA, NA, 19), cex = 0.85)
dev.off()

pdf("pred_results/rf_depth_flexibility.pdf", width = 9, height = 6)
plot(maxnodes_grid, depth_val_rmse, type = "b", pch = 19, col = "purple",
     ylim = range(c(depth_train_rmse, depth_val_rmse)),
     xlab = "Max Terminal Nodes (Tree Depth / Flexibility)", ylab = "RMSE",
     main = "Random Forest: Training vs. Validation RMSE across Tree Depth")
lines(maxnodes_grid, depth_train_rmse, type = "b", pch = 17, lty = 2, col = "darkgray")
points(best_maxnodes, min(depth_val_rmse), col = "red", pch = 19, cex = 1.5)
legend("topright", legend = c("Validation RMSE", "Training RMSE", paste("Optimal maxnodes =", best_maxnodes)),
       col = c("purple", "darkgray", "red"), lty = c(1, 2, NA), pch = c(19, 17, 19), cex = 0.85)
dev.off()

pdf("pred_results/rf_variable_importance.pdf", width = 10, height = 6)
varImpPlot(final_rf, main = "Tuned Random Forest Variable Importance", cex = 0.8)
dev.off()

pdf("pred_results/test_rmse_comparison.pdf", width = 10, height = 6)
par(mar = c(10, 4, 4, 2))
barplot(results_table$Test_RMSE, 
        names.arg = c("Baseline (Mean)", "Multiple Linear Reg.", "Forward Stepwise", "KNN Regression", "Tuned Random Forest"), 
        las = 2, col = "coral",
        ylab = "Test RMSE (Lower is Better)", main = "Comparison of Test RMSE Across Models",
        cex.names = 0.75, ylim = c(0, max(results_table$Test_RMSE) * 1.15))
abline(h = res_base["RMSE"], lty = 2, col = "black", lwd = 1.5)
legend("topright", legend = c("Baseline Benchmark RMSE"), lty = 2, col = "black", lwd = 1.5, cex = 0.8)
par(mar = c(5, 4, 4, 2))
dev.off()

cat("\n=== pred_model.R: All results and figures successfully exported to 'pred_results/' ===\n")