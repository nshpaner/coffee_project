# ==============================================================================
# Coffee Quality Inference Pipeline - inf_model.R
# Research Question: What is the partial effect of processing method and altitude on cup points?
# ==============================================================================

# Load necessary libraries
library(readr)
library(dplyr)
library(car) # Used for ANOVA and partial F-tests / partial R^2

# Step 1: Setup, Data Loading, and Preprocessing

coffee_ratings <- read_csv(here::here("data_info", "coffee_ratings.csv"))

# Select required variables for inference (treatment, outcome, confounders, mediators)
cols_to_use <- c("total_cup_points", "altitude_mean_meters", "processing_method", 
                 "country_of_origin", "species", "variety", 
                 "moisture", "category_one_defects", "category_two_defects")

df <- coffee_ratings[, cols_to_use]

# Impute missing values for categorical variables with "unknown"
cat_vars <- c("processing_method", "country_of_origin", "species", "variety")
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
numeric_vars <- c("total_cup_points", "altitude_mean_meters", "moisture", 
                  "category_one_defects", "category_two_defects")
for (var in numeric_vars) {
  df[[var]] <- as.numeric(df[[var]])
}

# Remove rows with missing values in required inference variables
df <- na.omit(df)

# Feature Engineering: Scale altitude per 100 meters for intuitive coefficient interpretation
df$altitude_100m <- df$altitude_mean_meters / 100

cat("=== STEP 1 COMPLETE: Cleaned Inference Dataset Dimensions ==\n")
print(dim(df))

# VISUALIZATION 1: Relationship between Altitude (100m) and Total Cup Points
plot(df$altitude_100m, df$total_cup_points, 
     main = "Coffee Quality vs. Growing Altitude (per 100m)",
     xlab = "Altitude (in hundreds of meters)", ylab = "Total Cup Points",
     col = adjustcolor("darkblue", alpha.f = 0.4), pch = 16)
abline(lm(total_cup_points ~ altitude_100m, data = df), col = "red", lwd = 2)
legend("topleft", legend = c("Coffee Lots", "OLS Trendline"), col = c("darkblue", "red"), pch = c(16, NA), lty = c(NA, 1))

# Step 2: Model 1 - Confounder-Adjusted Model (Total Effect)
# Controls for main confounders: Country, Species, and Variety

model1 <- lm(total_cup_points ~ processing_method + altitude_100m + 
               country_of_origin + species + variety, data = df)

# Extract coefficient summary with 95% Confidence Intervals
m1_summary <- summary(model1)
m1_ci <- confint(model1, level = 0.95)

# SUMMARY TABLE 1: Model 1 Key Coefficients (Processing & Altitude)
key_vars_m1 <- grep("processing_method|altitude_100m", names(coef(model1)), value = TRUE)
summary_table_m1 <- data.frame(
  Variable = key_vars_m1,
  Estimate = round(coef(model1)[key_vars_m1], 4),
  Std_Error = round(m1_summary$coefficients[key_vars_m1, "Std. Error"], 4),
  CI_Lower = round(m1_ci[key_vars_m1, 1], 4),
  CI_Upper = round(m1_ci[key_vars_m1, 2], 4),
  P_Value = format.pval(m1_summary$coefficients[key_vars_m1, "Pr(>|t|)"], digits = 3)
)

cat("\n=== SUMMARY TABLE 1: Model 1 (Confounder-Adjusted Total Effect) ===\n")
print(summary_table_m1)

# Step 3: Model 2 - Full Mediation Model (Direct Effect)
# Adds Mediators: Moisture and Defect Counts

model2 <- lm(total_cup_points ~ processing_method + altitude_100m + 
               country_of_origin + species + variety + 
               moisture + category_one_defects + category_two_defects, data = df)

m2_summary <- summary(model2)
m2_ci <- confint(model2, level = 0.95)

# SUMMARY TABLE 2: Model 2 Key Coefficients (Direct Effect)
key_vars_m2 <- grep("processing_method|altitude_100m", names(coef(model2)), value = TRUE)
summary_table_m2 <- data.frame(
  Variable = key_vars_m2,
  Estimate = round(coef(model2)[key_vars_m2], 4),
  Std_Error = round(m2_summary$coefficients[key_vars_m2, "Std. Error"], 4),
  CI_Lower = round(m2_ci[key_vars_m2, 1], 4),
  CI_Upper = round(m2_ci[key_vars_m2, 2], 4),
  P_Value = format.pval(m2_summary$coefficients[key_vars_m2, "Pr(>|t|)"], digits = 3)
)

cat("\n=== SUMMARY TABLE 2: Model 2 (Full Mediation Direct Effect) ===\n")
print(summary_table_m2)

# Step 4: Effect Size Evaluation via Partial R^2 (ANOVA)

# Compute Type II ANOVA to evaluate unique variance explained (Partial R^2)
anova_m2 <- Anova(model2, type = "II")
cat("\n=== ANOVA Table for Mediation Model (Model 2) ===\n")
print(anova_m2)

# Step 5: Comparative Inference Visualization

# VISUALIZATION 2: Coefficient Comparison Plot (Model 1 vs Model 2 for Altitude & Key Processing)
# Plotting confidence intervals for altitude coefficient across both models
alt_idx_m1 <- which(names(coef(model1)) == "altitude_100m")
alt_idx_m2 <- which(names(coef(model2)) == "altitude_100m")

est_m1 <- coef(model1)[alt_idx_m1]
ci_m1  <- m1_ci[alt_idx_m1, ]
est_m2 <- coef(model2)[alt_idx_m2]
ci_m2  <- m2_ci[alt_idx_m2, ]

par(mar = c(5, 6, 4, 2))
plot(c(est_m1, est_m2), c(2, 1), xlim = range(c(ci_m1, ci_m2)), ylim = c(0.5, 2.5),
     yaxt = "n", pch = 19, cex = 1.5, col = c("blue", "darkgreen"),
     xlab = "Estimated Effect per 100m Elevation Gain (Cup Points)", ylab = "",
     main = "Altitude Effect: Total vs. Direct Effect Comparison")
axis(2, at = c(1, 2), labels = c("Model 2 (Direct Effect)", "Model 1 (Total Effect)"), las = 1)

# Add 95% Confidence Interval error bars
segments(ci_m1[1], 2, ci_m1[2], 2, col = "blue", lwd = 2)
segments(ci_m2[1], 1, ci_m2[2], 1, col = "darkgreen", lwd = 2)
abline(v = 0, lty = 2, col = "gray")
legend("topright", legend = c("Model 1 (Confounders)", "Model 2 (+ Mediators)"), 
       col = c("blue", "darkgreen"), pch = 19, cex = 0.8)
par(mar = c(5, 4, 4, 2)) # Reset margins