# Coffee Quality Analysis, Prediction, and Inference

**Authors:** Nick Shpaner, Prerana Somarapu, Philip Valdehuesa  
**R Version:** 4.6.1 | **IDE:** RStudio  
*Note: This project is in-progress.*

\---

## Project Overview

Coffee is one of the world’s most traded agricultural commodities, yet its pricing and perceived quality are driven by a complex combination of environmental factors, post-harvest processing, bean characteristics, and sensory evaluations. Through this project, we apply data-driven and machine learning workflows to determine how to mathematically evaluate and predict coffee quality based on these variables.



This README outlines our foundational problem statement, translates it into separate prediction and inference research questions, and introduces our selected dataset (called `coffee\_ratings`) from the Coffee Quality Institute (CQI). For model evaluation, we reference the variable dictionary (`coffee\_ratings\_codebook`) located under `data\_info/`.

\---

## Platform-Independent Reproducibility \& Setup Guide

This project is built for **100% platform-independent reproducibility** across Windows, macOS, and Linux, leveraging native R projects (`.Rproj`) and the `here` package for robust, OS-agnostic path resolution.



### Getting Started

1. **Clone or Download** this repository to your local machine:

```bash
   git clone https://github.com/nshpaner/coffee\_project.git
   cd coffee\_project
   ```

2. Open **RStudio** (or your preferred R IDE).
3. Open the project by double-clicking **`coffee\_project.Rproj`** at the repository root. (This automatically sets your working directory to the project root).
4. Run the dependency installation script (first code chunk of `npp\_coffee\_project.nb`) to automatically install required packages (such as `here`, `readr`, `dplyr`, `caret`, `leaps`, `FNN`, `randomForest`, and `car`):

```R
   source("code+nb/install\_deps.R")
   ```

\---

## Repository Structure

```text
coffee\_project/
├── coffee\_project.Rproj        # R Project root anchor file
├── install\_deps.R              # Dependency installation and verification script
├── install\_deps\_README.txt     # Dependency guide and instructions
├── data\_info/                  
│   ├── coffee\_ratings.csv      # Raw coffee cupping dataset
│   └── coffee\_ratings\_codebook # Variable dictionary codebook
└── code+nb/                    
    ├── install\_deps.R          # Subfolder reference copy of dependency script
    ├── npp\_coffee\_project\_nb.qmd # Quarto interactive execution notebook
    ├── pred\_model.R            # Predictive modeling and machine learning pipeline
    └── inf\_model.R             # Statistical inference and regression analysis script
```

\---

## Problem Statement, Research Questions, and Metrics

The specialty coffee market operates on strict quality thresholds where subtle differences in the overall cup score significantly alter a farm’s financial return. Coffees scoring 80 points or higher on the 100-point CQI scale qualify as "specialty coffee," commanding higher brand recognition and premium pricing, while coffees below 80 are categorized as commodity grade. Assessing coffee quality through formal tasting panels is labor-intensive, subjective, and costly, and producers often lack quantitative guidance on how agricultural techniques impact sensory profiles.

To address this, we define two complementary research questions:



### 1\. Prediction:

> Can we learn a function $Y = f(\\mathbf{X})$ to accurately predict a coffee lot’s overall quality score ($\\text{total\\\_cup\\\_points}$) using only its non-sensory physical, agronomic, and geographic attributes ($\\mathbf{X}$)?

* **Mathematical Context:** $Y$ represents $\\text{total\_cup\_points}$ (a continuous score from 0 to 100). The predictor feature set $\\mathbf{X}$ includes physical bean characteristics (moisture, category-one/two defects, color), geographic attributes (altitude, country of origin), and agronomic parameters (species, variety, processing method).
* **Performance Metrics:** The primary metric is Root Mean Squared Error ($\\text{RMSE}$), targeting an $\\text{RMSE} \\le 1.25$ cup points on out-of-sample test data (within typical professional cupping panel tolerance). Secondary metrics include Mean Absolute Error ($\\text{MAE}$) and Coefficient of Determination ($R^2 \\ge 0.45$).



### 2\. Inference:

> What is the partial effect of post-harvest processing method ($\\text{processing\\\_method}$) and growing elevation ($\\text{altitude\\\_mean\\\_meters}$) on overall sensory quality ($\\text{total\\\_cup\\\_points}$), when controlling for bean species, moisture level, physical defect counts, and country of origin?

* **Mathematical Context:** Formulated via a multiple linear regression model:
$$\\text{total\_cup\_points} = \\beta\_0 + \\beta\_1 (\\text{processing\_method}) + \\beta\_2 (\\text{altitude\_mean\_meters}) + \\boldsymbol{\\beta}*{\\text{controls}}\\mathbf{X}*{\\text{controls}} + \\epsilon$$
* **Hypothesis Testing:** Two-tailed $t$-tests on regression coefficients with a significance threshold of $\\alpha = 0.05$ and $95%$ confidence intervals. Effect sizes are measured via partial $R^2$.

\---

## Dataset Selection \& Introduction

We selected the Coffee Quality Institute (CQI) Specialty Coffee Dataset, aggregated through the CQI database in 2018 and compiled by James LeDoux in 2020, later refined via the R TidyTuesday open-data initiative.

The dataset contains 1,339 rows and 27 total columns (we selected 14 active variables for analysis). Each row represents a commercial lot of green coffee beans submitted to an accredited CQI in-country partner for formal Q-Grading certification.

\---

## Potential Obstacles \& Mitigation Strategies

1. **Small \& Unbalanced Groups:** Rare coffee varieties (e.g., Robusta, pulped natural/honey) have very few observations. *Mitigation:* Group countries with fewer than 20 lots into an "Other" category and limit comparative claims.
2. **Unreliable/Missing Self-Reported Data:** Farm-reported attributes like altitude and moisture contain noise and errors. *Mitigation:* Exclude extreme outliers into an "unknown" category and perform robustness checks.
3. **Weak Predictive Signal:** Non-sensory features capture a smaller share of score variance ($R^2 \\approx 0.15 - 0.25$). *Mitigation:* Compare flexible machine learning models against a naive baseline (predicting the mean).

\---

## Script Overviews

* **`install\_deps.R`**  
Checks the R library against requirements, installs missing packages from CRAN with single-threaded safety (`Ncpus = 1`), and ensures all modeling libraries are ready.



* **`pred\_model.R`** (Predictive Modeling)  
Ingests the dataset using `here::here()`, handles preprocessing, trains machine learning models (KNN, Random Forest), and evaluates out-of-sample performance.



* **`inf\_model.R`** (Statistical Inference)  
Fits regression models, evaluates coefficient significance and multicollinearity using the `car` package, and tests hypotheses regarding coffee cupping attributes.



* **`npp\_coffee\_project\_nb.qmd`** (Interactive Notebook)  
An integrated Quarto report located in `code+nb/` configured with global `root.dir` chunks anchored via `here::here()` to run the pipeline, render visualizations, and compile the final report.

\---

## AI Usage Statement

We used AI to assist in debugging code and clarifying technical concepts. All data interpretations, analytical conclusions, and write-ups are our original work.

\---

## References

* Coffee Quality Institute (CQI). Database of Certified Quality Evaluations. CQI Public Registry, 2018.
* LeDoux, J. (2018). Coffee Quality Database. GitHub repository: [jldbc/coffee-quality-database](https://github.com/jldbc/coffee-quality-database).

