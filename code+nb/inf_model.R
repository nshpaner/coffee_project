# ==============================================================================
# Coffee Quality Inference Pipeline Script

# Research Question: What is the partial effect of processing method and
# altitude on total cup points?
# ==============================================================================

library(readr)
library(dplyr)
library(car) # Type II ANOVA and Generalized VIF (GVIF)

# Step 1: Data Loading & Cleaning
# ==============================================================================
# Our first step loads the coffee ratings dataset, cleans missing records, 
# scales the altitude per 100 meters, and helps align factor levels with a 
# function to control for small category sizes
 
# This makes sure we have very intuitive coefficient scaling 
# (per 100m elevation gain) and accounts for unobserved categorical 
# levels across subsets so that the levels map cleanly to "other" to
# avoid design matrix crashes.
# ==============================================================================

coffee_ratings <- read_csv(here::here("data_info", "coffee_ratings.csv"), show_col_types = FALSE)

cols_to_use <- c("total_cup_points", "altitude_mean_meters", "processing_method", 
                 "country_of_origin", "species", "variety", 
                 "moisture", "category_one_defects", "category_two_defects")

df <- coffee_ratings[, cols_to_use]

cat_vars <- c("processing_method", "country_of_origin", "species", "variety")
for (var in cat_vars) {
  df[[var]] <- as.character(df[[var]])
  df[[var]][is.na(df[[var]]) | df[[var]] == "" | df[[var]] == "NA"] <- "unknown"
  df[[var]] <- as.factor(df[[var]])
}

# Group low-frequency categories (< 5 lots) into "other" across the full dataset
for (var in cat_vars) {
  counts <- table(df[[var]])
  rare_cats <- names(counts[counts < 5])
  levels_df <- levels(df[[var]])
  levels_df[levels_df %in% rare_cats] <- "other"
  levels(df[[var]]) <- factor(levels_df)
}

# Safe factor level harmonizer to prevent "has new levels" errors
align_factor_levels <- function(train_df, new_df, cat_vars) {
  for (var in cat_vars) {
    train_levels <- levels(train_df[[var]])
    new_df[[var]] <- as.character(new_df[[var]])
    new_df[[var]][!new_df[[var]] %in% train_levels] <- "other"
    new_df[[var]] <- factor(new_df[[var]], levels = train_levels)
  }
  return(new_df)
}

numeric_vars <- c("total_cup_points", "altitude_mean_meters", "moisture", 
                  "category_one_defects", "category_two_defects")
for (var in numeric_vars) {
  df[[var]] <- as.numeric(df[[var]])
}

df <- na.omit(df)
df <- df[df$total_cup_points > 0, ]

# Apply alignment to make sure the design matrices hold up
df <- align_factor_levels(df, df, cat_vars)

# Scale altitude per 100 meters for interpretable effect sizes
df$altitude_100m <- df$altitude_mean_meters / 100

cat("=== STEP 1 COMPLETE: Cleaned Inference Dataset Dimensions ===\n")
print(dim(df))

# Graph 1: Bivariate Relationship (this is the unadjusted benchmark)
generate_bivariate_plot <- function() {
  plot(df$altitude_100m, df$total_cup_points, 
       main = "Coffee Quality vs. Growing Altitude (per 100m)",
       xlab = "Altitude (in hundreds of meters)", ylab = "Total Cup Points",
       col = adjustcolor("darkblue", alpha.f = 0.4), pch = 16)
  abline(lm(total_cup_points ~ altitude_100m, data = df), col = "red", lwd = 2)
  legend("topleft", legend = c("Coffee Lots", "Unadjusted OLS Trendline"), 
         col = c("darkblue", "red"), pch = c(16, NA), lty = c(NA, 1), cex = 0.85)
}

if (!dir.exists("inf_results")) dir.create("inf_results")

if (interactive()) {
  generate_bivariate_plot()
}

pdf("inf_results/altitude_bivariate_trend.pdf", width = 9, height = 6)
generate_bivariate_plot()
dev.off()

# Step 2: Nested Model Progression (Baseline -> Confounder-Adjusted -> Mediation)
# ==============================================================================
# Now, we will fit a nested sequence of linear models: 

# Model 0 (unadjusted)
# Model 1 (confounder-adjusted total effect)
# Model 2 (full mediation model; allows us to see direct effect)

# We do this to isolate the actual partial effect of altitude_100m and  
# processing_method by systematically controlling for potential geographic
# and botanical confounding variables (as identified in Memo 2).
# ==============================================================================

# Model 0
model0 <- lm(total_cup_points ~ processing_method + altitude_100m, data = df)

# Model 1
model1 <- lm(total_cup_points ~ processing_method + altitude_100m + 
               country_of_origin + species + variety, data = df)

# Model 2
model2 <- lm(total_cup_points ~ processing_method + altitude_100m + 
               country_of_origin + species + variety + 
               moisture + category_one_defects + category_two_defects, data = df)

extract_key_table <- function(mod) {
  s  <- summary(mod)
  ci <- confint(mod, level = 0.95)
  kv <- grep("processing_method|altitude_100m", names(coef(mod)), value = TRUE)
  data.frame(
    Variable  = kv,
    Estimate  = round(coef(mod)[kv], 4),
    Std_Error = round(s$coefficients[kv, "Std. Error"], 4),
    CI_Lower  = round(ci[kv, 1], 4),
    CI_Upper  = round(ci[kv, 2], 4),
    P_Value   = format.pval(s$coefficients[kv, "Pr(>|t|)"], digits = 3),
    row.names = NULL
  )
}

cat("\n=== SUMMARY TABLE 1: Model 1 (Confounder-Adjusted Total Effect) ===\n")
print(extract_key_table(model1))

cat("\n=== SUMMARY TABLE 2: Model 2 (Full Mediation Direct Effect) ===\n")
print(extract_key_table(model2))

# Step 3: Overfitting, Generalization & Preventing Multicollinearity
# ==============================================================================
# This next step involves computing 10-fold cross-validation RMSE, AIC/BIC, 
# adjusted R^2, and Generalized Variance Inflation Factors (GVIF).

# We do this to empirically confirm that adding controls and mediators 
# improves the fit to unseen data, rather than overfitting to random noise.
# In doing so, we make sure that the standard errors aren't inflated by 
# severe multicollinearity.

# Due to the limited non-sensory categorical values we have, trying to drill down 
# related predictors and validate any interpreted effects is of utmost importance 
# for an inference model.
# ==============================================================================

# 10-Fold Cross-Validation RMSE
calc_cv_rmse <- function(formula, data, k = 10) {
  set.seed(42)
  folds <- sample(rep(1:k, length.out = nrow(data)))
  sq_errs <- numeric(nrow(data))
  
  for (fold in 1:k) {
    tr <- data[folds != fold, ]
    te <- data[folds == fold, ]
    
    # Collapse any category with < 2 observations in the training fold into "other"
    for (var in names(tr)) {
      if (is.factor(tr[[var]])) {
        tr_counts <- table(tr[[var]])
        valid_lvls <- names(tr_counts[tr_counts >= 2])
        
        tr[[var]] <- as.character(tr[[var]])
        tr[[var]][!tr[[var]] %in% valid_lvls] <- "other"
        tr[[var]] <- factor(tr[[var]])
        
        te[[var]] <- as.character(te[[var]])
        te[[var]][!te[[var]] %in% levels(tr[[var]])] <- "other"
        te[[var]] <- factor(te[[var]], levels = levels(tr[[var]]))
      }
    }
    
    fit <- lm(formula, data = tr)
    preds <- predict(fit, newdata = te)
    sq_errs[folds == fold] <- (te$total_cup_points - preds)^2
  }
  sqrt(mean(sq_errs))
}

diag_table <- data.frame(
  Model          = c("Model 0 (Unadjusted)", "Model 1 (+ Confounders)", "Model 2 (+ Mediators)"),
  Num_Params     = c(length(coef(model0)), length(coef(model1)), length(coef(model2))),
  Adj_R2         = round(c(summary(model0)$adj.r.squared, summary(model1)$adj.r.squared, summary(model2)$adj.r.squared), 4),
  AIC            = round(c(AIC(model0), AIC(model1), AIC(model2)), 1),
  BIC            = round(c(BIC(model0), BIC(model1), BIC(model2)), 1),
  CV_10Fold_RMSE = round(c(
    calc_cv_rmse(formula(model0), df),
    calc_cv_rmse(formula(model1), df),
    calc_cv_rmse(formula(model2), df)
  ), 4)
)

cat("\n=== SUMMARY TABLE 3: Model Complexity & Overfitting Outlook ===\n")
print(diag_table)

# Multicollinearity Check (GVIF)
cat("\n=== Generalized Variance Inflation Factors (Model 2) ===\n")
print(round(vif(model2), 3))

# Type II ANOVA for Partial Variance Explained
cat("\n=== Type II ANOVA Table (Model 2) ===\n")
print(Anova(model2, type = "II"))

# Step 4: Bootstrapped Check for the Effect of Altitude (B = 500)
# ==============================================================================
# This leg of the script performs bootstrap resampling (500 data points), which
# helps with tracking the distribution of the altitude coefficient.

# Bootstrapping validates the stability of the altitude_100m coefficient and 
# proves that our inference conclusions are robust to natural sampling variation
# and not some sort of one-off from arbitrarily skewed data.
# ==============================================================================

set.seed(42)
B <- 500
boot_alt_m2 <- numeric(B)
for (b in 1:B) {
  boot_idx <- sample(1:nrow(df), replace = TRUE)
  boot_fit <- lm(formula(model2), data = df[boot_idx, ])
  boot_alt_m2[b] <- coef(boot_fit)["altitude_100m"]
}

cat("\n=== Bootstrap Stability Check (Altitude 100m Direct Effect, B = 500) ===\n")
cat("OLS Estimate:", round(coef(model2)["altitude_100m"], 4),
    "| Bootstrap Mean:", round(mean(boot_alt_m2), 4),
    "| OLS SE:", round(summary(model2)$coefficients["altitude_100m", "Std. Error"], 4),
    "| Bootstrap SE:", round(sd(boot_alt_m2), 4), "\n")

# Step 5: Comparative Coefficient Plot Across Models 0, 1, and 2
# ==============================================================================
# We've made it to the final step of the script! We will generate a coefficient 
# comparison visualization that shows point estimates and 95% confidence 
# intervals for altitude_100m across Models 0, 1, and 2.

# This graph helps communicate how controlling for confounding variables 
# is able to shifts the estimated effect size. It serves as an intuitive,
# empirically-based answer for our research question.
# ==============================================================================

generate_stability_plot <- function() {
  models_list <- list(model0, model1, model2)
  ests <- sapply(models_list, function(m) coef(m)["altitude_100m"])
  cis  <- sapply(models_list, function(m) confint(m)["altitude_100m", ])
  
  par(mar = c(5, 11, 4, 2))
  plot(ests, 3:1, xlim = range(c(cis, 0)), ylim = c(0.5, 3.5),
       yaxt = "n", pch = 19, cex = 1.5, col = c("darkred", "blue", "darkgreen"),
       xlab = "Estimated Effect per 100m Elevation Gain (Cup Points)", ylab = "",
       main = "Altitude Effect Stability Across Model Specifications")
  axis(2, at = 3:1, labels = c("Model 0 (Unadjusted)", "Model 1 (+ Confounders)", "Model 2 (+ Mediators)"), las = 1)
  segments(cis[1, ], 3:1, cis[2, ], 3:1, col = c("darkred", "blue", "darkgreen"), lwd = 2)
  abline(v = 0, lty = 2, col = "gray40")
  par(mar = c(5, 4, 4, 2))
}

if (interactive()) {
  generate_stability_plot()
}

pdf("inf_results/altitude_effect_stability.pdf", width = 10, height = 6)
generate_stability_plot()
dev.off()

# ==============================================================================
# EXPORT & REPORTING: Save Inference Result Summary Tables to inf_results/
# ==============================================================================

# Create output directory if it doesn't exist
if (!dir.exists("inf_results")) {
  dir.create("inf_results")
}

# Save Summary Tables as CSV files
write.csv(extract_key_table(model1), "inf_results/model1_confounder_adjusted.csv", row.names = FALSE)
write.csv(extract_key_table(model2), "inf_results/model2_mediation_effect.csv", row.names = FALSE)
write.csv(diag_table, "inf_results/model_diagnostics_table.csv", row.names = FALSE)

cat("\n=== inf_model.R: All results and figures successfully exported to 'inf_results/' ===\n")