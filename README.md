Machine Learning for house pricing prediction in Ames, Iowa.

New York City Data Science Academy
TTP-MachineLearning-Project
Group : Ira Villar, Bee Kim, Drucila LeFevre, Tomas Nivon

Overview
--------
Using machine learning algorithms, this project was conducted to predict the sales prices of houses in the city of Ames, Iowa. 

DataSet
-------
Collected from a Kaggle Competitiion which consisted of 80 variables. 
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Background Information 
----------
City: Ames, Iowa
80 Variables
Years: 2006 - 2010
Train and Test Set
Root Mean Squared Error (RMSE) between predicted value and observed sales price

Understanding the Data
----
House Sale Price
--
- The cheapest house was sold for $34,900 and the most expensive for $755,000
- The average sale price is $180,921, while median is $163,000
- Number of houses were sold in June

Cleaning the Data
----
Took the logarithmic of Sales Price : reduce the sknewness 
Outlier Detection: removed outliers to prevent bias in our results
Missing Values: 
- Removed columns due to over 80% missingness
- Given that some missing values had meaning within them, we imputed these columns
- Skewness Numerical Variables with sknewness > 0 : Box Cox Method
- Dummy Variables: transformed categorical variables into dummy variables

Feature Engineering 
----
Variables: 
Total Square Feet
Total Number of Bathrooms
Total Square Feet of the Porch
Ratio of above ground square feet to total square feet
Percentage of finished basement over total basement
Years between year remodeled and year built
Years between year sold and year remodeled
Neighborhood dummy variable * total square foot
Neighborhood dummy variable * overall quality

Machine Learning Models 
---

Linear: OLS, LASSO, ELASTIC NET
NonLinear: Random Forest, XGBoost
Stacked
Procedure: 
- Preprocessing 
- Fit Linear Models, Tune hyperparameters (Grid SearchCV), compare train-test RSME
- Test different combinations of conditions in preprocessing
- Fit more models (RF, XGB)
- Average models

Feature Importance
---
Common Features across models: Overall Quality, Total Square Foot, Year Built

Conclusion 
---
- Lasso produced the best result
- More advanced models did not out perform the linear regression models 

Kaggle Scores
---
LASSO: 0.12104
OLS : 0.12215
ELASTIC NET: 0.12611
RANDOM FOREST: 0.1441
XGB: 0.12787
STACKED: 0.124

Future Improvement 
--
- Non- linear models without imputations and dummy variables
- Seasonality : time series
- Combine similar neighborhoods and do imputations based on Neighborhood 


