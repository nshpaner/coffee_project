# ==============================================================================
# Coffee Project Pipeline - Dependency Installation Script
# ==============================================================================
# Safely checks and installs all required packages for data loading,
# preprocessing, cross-validation, machine learning, and statistical inference.
# Includes 'here' for portable, platform-independent project path resolution.
# ==============================================================================

required_packages <- c(
  "here",         # Portable, OS-independent root path resolution
  "readr",        # For fast, robust CSV and text file loading
  "dplyr",        # Data manipulation and cleaning pipelines
  "caret",        # Machine learning modeling tools & pre-processing utilities
  "leaps",        # Best subset and forward/backward stepwise regression selection
  "FNN",          # Fast K-Nearest Neighbors regression implementation
  "randomForest", # Random Forest ensemble modeling
  "car"           # Companion to Applied Regression (ANOVA, partial F-tests, partial R^2)
)

# Identify which required packages are missing from the current library
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

# Install missing packages sequentially to prevent background worker crashes and ensure portability
if (length(new_packages) > 0) {
  message("Installing missing packages: ", paste(new_packages, collapse = ", "))
  
  install.packages(
    new_packages, 
    repos = "https://cloud.r-project.org",
    dependencies = TRUE,
    type = "both",
    Ncpus = 1 # Forces single-threaded execution to avoid worker thread dropouts
  )
  
  message("Package installation complete!")
} else {
  message("All required packages are already installed and up to date!")
}