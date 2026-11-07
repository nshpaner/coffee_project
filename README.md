# Coffee Cupping Score Predictive \& Inferential Modeling Project

This repository contains an end-to-end R analysis pipeline for exploring, modeling, and evaluating coffee quality metrics using a comprehensive coffee cupping score dataset (`coffee\_ratings.csv`).

The project is built for **100% platform-independent reproducibility** across Windows, macOS, and Linux, leveraging native R projects (`.Rproj`) and the `here` package for robust, OS-agnostic path resolution.

\---

## Repository Structure

```text
coffee\_project/
├── coffee\_project.Rproj        # R Project root anchor file
├── renv.lock                   # Package version control lockfile
├── install\_deps.R              # Dependency installation and verification script
├── data\_info/                  
│   └── coffee\_ratings.csv      # Raw coffee cupping dataset
└── code+nb/                    
    ├── install\_deps.R          # Subfolder reference copy of dependency script
    ├── analysis\_notebook.qmd   # Quarto / R Markdown interactive execution notebook
    ├── pred\_model.R            # Predictive modeling and machine learning pipeline
    └── inf\_model.R             # Statistical inference and regression analysis script
```

\---

## Getting Started \& Setup Guide

To run this project on any machine without path translation or environment errors, follow these steps:

1. **Clone or Download** this repository to your local machine.
2. Open **RStudio** (or your preferred R IDE).
3. Open the project by double-clicking **`coffee\_project.Rproj`** at the root of the repository. (This automatically sets your working directory to the project root).
4. Run initial setup code chunk to activate the renv and the dependency installation script in your R console to automatically install any required packages (such as `here`, `readr`, `dplyr`, `caret`, `leaps`, `FNN`, `randomForest`, and `car`):

```R
   source(here("code+nb/install\_deps.R") # After library(here) is ran
   ```

\---

## Script Overviews

* **`install\_deps.R`**
Checks your current R library against the project's dependency requirements, automatically downloads and installs any missing packages from CRAN with single-threaded safety (`Ncpus = 1`), and ensures all modeling and data-munging libraries are ready.



* **`pred\_model.R`** (Predictive Modeling)
Focuses on machine learning and predictive performance. It ingests the coffee ratings dataset using `here::here()`, performs data pre-processing and feature selection, trains predictive models (such as KNN and Random Forest), and evaluates out-of-sample performance metrics.



* **`inf\_model.R`** (Statistical Inference)
Focuses on explanatory modeling and statistical inference. It fits regression models, evaluates coefficient significance and multicollinearity using the `car` package, and tests hypotheses regarding coffee cupping attributes.



* **`npp\_coffee\_project\_nb.qmd`** (Notebook)
An integrated Quarto / R Markdown report located in the `code+nb` subfolder. It incorporates global `root.dir` chunk options anchored via `here::here()` to seamlessly execute the pipeline, render visualizations, and compile the final analytical report. This is where ALL scripts should be ran to produce all modeling and visualizations.

