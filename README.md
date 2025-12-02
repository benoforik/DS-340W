Overview

This project builds an end-to-end weekly injury-prediction system using data from Transfermarkt.
It constructs detailed player-week timelines, engineers injury-history features, trains multiple ML models, and visualizes long-term injury trends.

The pipeline predicts:

“Will a player get injured next week?”

It also analyzes:

Changes in injury frequency since 2020

How risk evolves with recovery time

📂 Project Structure
📁 project/
│── README.md
│── transfermarkt_new.csv      # Raw data for ML modeling
│── transfermarkt.csv          # Full dataset for trend analysis
│── final_project.Rmd          # Full analysis and modeling pipeline
│── plots/                     # Generated graphs (optional)
└── models/                    # Saved models (optional)

Key Features
 1. Data Cleaning & Standardization

Parse injury dates correctly (multiple formats supported)

Convert duration into numeric values

Remove malformed entries

Sort injury sequences per player

 2. Weekly Player Timeline Generation

Creates a complete grid:

player × every week between first injury and last injury


This ensures consistent time-series modeling, even with no injury events.

 3. Feature Engineering

For each player-week:

injury_this_week

past_injuries (cumulative)

last_injury_week (forward filled)

weeks_since_last_injury (recovery time)

injury_next_week (prediction target)

 4. Chronological Train/Test/Validation Split

Prevents data leakage:

70% → training

20% → testing

10% → validation

 5. Machine Learning Models
Model	Purpose
Logistic Regression	Simple linear baseline
Random Forest	Nonlinear, interpretable, robust
XGBoost	Handles imbalance well, high performance

Metrics calculated:

Confusion Matrix

Precision & Recall

F1 Score

PR-AUC (best for rare-event prediction)

 6. Trend Analysis (2020–Present)

Plots number of muscular/ligament injuries per season to detect:

Are injuries increasing over time?

 7. Risk Curve: Time Since Last Injury → Injury Probability

Visualizes short-term re-injury risk for the first 20 weeks after return.

 Example Outputs
Injury Trend Since 2020

 Line plot showing injury counts by season.

Re-Injury Risk Curve

 Probability of injury vs. weeks since last injury.

Model Evaluation

 Precision, Recall, F1, and PR-AUC comparisons across models.

 Technologies Used
Category	Tools
Data Wrangling	tidyverse, lubridate, tidyr
Machine Learning	ranger, xgboost, randomForest
Evaluation	caret, PRROC
Visualization	ggplot2
Reproducibility	R Markdown
▶️ How to Run the Project

Clone the repository

Ensure the CSV files are in your working directory

Open the .Rmd or .R file in RStudio

Install dependencies:

install.packages(c("tidyverse", "lubridate", "randomForest", 
                   "PRROC", "caret", "xgboost", "ranger"))


Run the script top-to-bottom

View generated plots and model performance outputs

 Use Cases

Player workload monitoring

Injury prevention analytics

Training & rehabilitation optimization

Research in sports science

Predictive modeling pipeline for athlete health

 Notes

Injury events are rare, so PR-AUC is the best metric.

Weeks with no prior injury use 999 as placeholder recovery time.

Trend analysis uses season strings like "20/21", converted to numeric.

The dataset is inherently imbalanced — XGBoost mitigates this with scale_pos_weight.

 Summary

This repository provides a fully reproducible:

Injury forecasting system

Weekly player-timeline builder

ML comparison framework

Trend analysis dashboard

Re-injury risk quantification

It demonstrates how machine learning and data science can support athlete health and sports performance decisions.
