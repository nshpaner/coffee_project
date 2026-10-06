\# Coffee Quality Analysis, Prediction, and Inference

\# Authors: Nick Shpaner, Prerana Somarapu, Philip Valdehuesa

\# R Version: 4.5.2 | IDE: RStudio

NOTE: This project is still in-progress

\# Project Overview

Coffee is one of the world’s most traded agricultural commodities, yet its pricing and perceived quality are driven by a complex combination of environmental factors, post-harvest processing, characteristics of the beans, and people’s sensory evaluations. Through this project, our group is attempting to use data-driven and machine learning processes to determine how to mathematically evaluate and predict coffee quality, based on a number of these variables.


This README outlines our foundational problem statement, translates it into separate prediction and inference research questions, and introduces our selected dataset (called “coffee\_ratings”) from the Coffee Quality Institute (CQI) in detail. For the purpose of evaluating our statistical models, we use a detailed variable dictionary (see coffee\_ratings\_codebook under data\_info) for all the data features used in subsequent modeling phases.


\# Problem Statement, Research Questions, and Metrics


The specialty coffee market operates on strict quality thresholds where subtle differences in the overall cup score can significantly alter a farm’s financial return. Coffees that score 80 points or higher on the 100-point Coffee Quality Institute (CQI) scale qualify as “specialty coffee”, which brings forth higher brand recognition and more expensive products. Meanwhile, coffees below 80 are categorized as commodity grade. However, assessing coffee quality through formal tasting panels is labor-intensive, subjective, and costly. Plus, producers in developing regions generally don’t have clear, quantitative guidance on how specific agricultural techniques like processing or managing elevation is able to directly impact the beans’ sensory profiles and their overall marketability. A major challenge in the coffee industry is understanding how objective pre-harvest and post-harvest physical features translate into a multifaceted understanding of a bean’s sensory quality, and whether overall cup scores can be accurately predicted before a formal Q-grader evaluation takes place.


There are numerous stakeholders that stand to benefit from our group’s findings. For starters, smallholder coffee farmers make decisions based on resource availability and their processing infrastructure. An example of this would be whether they should invest in washed or natural processing equipment. Farmers can use quantitative insights from our study to see the effect of processing and elevation, and how they can optimize the structure and management of their farms to maximize their bean quality. Additionally, roasters and green coffee import businesses would find a lot of utility in our research. Purchasing managers need objective predictive systems in order to conjure generalizable estimates for bean quality. Plus, they would need a model to help them target any undervalued coffee lots based on physical and geographic attributes before international shipping. Finally, certifiers and coffee analysts would benefit from understanding the empirical statistical relationships that may exist between potential physical defects, moisture levels, and multivariate sensory scores (aftertaste, aroma, etc). Being able to quantifiably understand these potential relationships helps build and maintain highly standardized, strict, grading criteria used to evaluate coffee from different businesses.


To be able to predict coffee quality, and understand the mechanisms that influence it, we define two complementary research questions (including variables from the dataset):


\## Prediction:

Can we learn a function Y = f(X) to accurately predict a coffee lot’s overall quality score (total\_cup\_points) using only its non-sensory physical, agronomic, and geographic attributes (X)?


Looking at it through a mathematical context, Y represents total\_cup\_points (a continuous score from 0 to 100). The predictor feature set X includes physical bean characteristics (moisture, category\_one\_defects, category\_two\_defects, color), geographic attributes (altitude\_mean\_meters, country\_of\_origin), and agronomic parameters (species, variety, processing\_method). As it relates to the problem statement, predicting the overall cup quality only from objective physical and geographic features allows potential roasters and importers to check green coffee lots before committing to physical sampling and cupping panels. In doing so, these stakeholders can streamline their evaluation process by filtering down to objective features before greenlighting the beans for collection. The main metric we plan to use for this research question would be the Root Mean Squared Error (RMSE). We set a performance target of RMSE <= 1.25 cup points on out-of-sample test data. Given that the standard deviation of total\_cup\_points in the dataset is 3.50 points (and the median score is 82.50), an RMSE under 1.25 points makes sure that all the predictions fall within the typical tolerance of professional cupping panels (which is +- 1.5 points). Some secondary, potential backup metrics we can look at would be the Mean Absolute Error (MAE) and Coefficient of Determination (R^2) for total\_cup\_score, with a target R^2 >= 0.45, with the goal of explaining the variation in total\_cup\_score in the dataset.


\## Inference:

What is the partial effect of post-harvest processing method (processing\_method) and the growing elevation (altitude\_mean\_meters) on the overall sensory quality (total\_cup\_points), when controlling for bean species, moisture level, physical defect counts, and country of origin?


Through a mathematical context, we used the variables from the dataset to formulate a multiple linear regression model:


total\_cup\_points = B0 + B1 (processing\_method) + B2 (altitude\_mean\_meters) + (upsilon)X\_controls + epsilon


where X\_controls include species, moisture, category\_one\_defects, category\_two\_defects, and country\_of\_origin. As this relates to our problem statement, isolating the effect of post-harvest processing methods (like Washed vs. Natural/Dry) and the growing altitude provides actionable information for farmers. Farmers can use the data to decide whether to use specific post-harvest methods or plant the beans at higher elevations. Pertaining to point estimates (B\_hat), the estimated score differences for each processing method would be relative to the baseline (Washed/Wet, still haven’t decided). The estimated score changes per 100-meter gain in elevation. For the purposes of hypothesis testing, two-tailed t-tests on regression coefficients will be used with a pre-set significance threshold of alpha = 0.05. The confidence intervals will be pre-set at 95% for and B\_hat\_processing and B\_hat\_altitude. A successful answer will have tight confidence intervals (for example, the width would be < 0.8 cup points for different processing types), which would help inform clear policy conclusions for potential producers. The effect size would be measured via partial R^2 to measure the proportion of variance that’s uniquely explained by post-harvest processing and altitude.


\# Dataset Selection and Introduction

We selected the Coffee Quality Institute (CQI) Specialty Coffee Dataset, originally aggregated and made public through the CQI database in 2018, and compiled by James LeDoux in 2020. The dataset was later refined and curated for academic use via the R TidyTuesday open-data initiative.


\## References

Coffee Quality Institute (CQI). Database of Certified Quality Evaluations. CQI Public Registry, 2018.

LeDoux, J. (2018). Coffee Quality Database. GitHub repository: jldxc/coffee\_quality\_database

https://github.com/jldbc/coffee-quality-database


The dataset contains 1,339 rows (> 500) and has 27 total columns. We selected 14 active variables for analysis (excluding the row index rownames), satisfying the requirement of at least 10 active variables. A single data point (aka an individual row) in this dataset represents one commercial lot of green coffee beans submitted to an accredited CQI in-country partner for a formal Q-Grading certification. Each of the lots are evaluated by a panel of licensed Q Graders who take note of the physical bean metrics, farm origin metadata, and standardized sensory ratings.


For information on the dataset variables, see coffee\_ratings\_codebook under the data\_info in the repo.


\# Installation

git clone https://github.com/nshpaner/coffee\_project.git
cd coffee\_project