========================================================================
Coffee Quality Prediction Project: Dependency Installation Instructions
========================================================================

Packages Included:
- here          : Portable, OS-independent root path resolution
- readr         : Data ingestion
- dplyr         : Data transformation & cleaning
- caret         : Model training utilities
- leaps         : Stepwise & subset regression
- FNN           : K-Nearest Neighbors (KNN) regression
- randomForest  : Random Forest modeling
- car           : Companion to Applied Regression (ANOVA & multicollinearity)

------------------------------------------------------------------------
Recommended Workflow for Execution & Reproducibility
------------------------------------------------------------------------

1. Project Setup:

   Always open the project by double-clicking the `coffee_project.Rproj` file 
   at the repository root. This automatically locks your working directory to 
   the project root regardless of operating system (Windows, macOS, or Linux).

2. Recommended External Run:

   Run the `install_deps.R` script ONCE in your RStudio console prior to 
   running your model scripts or rendering your Quarto / R Markdown notebook:

   In RStudio Console (from root):
   > source("install_deps.R")

   If running from a subfolder (like code+nb/):
   > source(here::here("install_deps.R"))

3. DO NOT embed install.packages() directly inside other scripts:

   - Running install.packages() inside analysis scripts slows down execution 
     every time you knit or run your code.
   - It can overwrite existing package configurations on collaborators' machines.
   - Keeping dependencies in a separate script ensures clean, modular execution.

4. Top of scripts and notebooks (`pred_model.R`, `inf_model.R`, notebooks):
   Initialize `here` and load your required libraries cleanly:

   library(here)
   library(readr)
   library(dplyr)
   library(caret)
   library(leaps)
   library(FNN)
   library(randomForest)
   library(car)