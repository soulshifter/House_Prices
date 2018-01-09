# House Prices
House Prices Kaggle Problem

# DATA:
train.csv : Training set
test.csv : Test set
output.csv : Final output file for regression task
sample_submission.csv : Sample .csv file for submission
data_description.txt : Attribute description

# LOGIC USED :
1) Checking data shape and description both for test and train set.
2) Fixing the target i.e. "SalePrice".
3) Combining train and test set for easier data handling.
4) For data preprocessing :
   -> Checking numerical and categorical features.
   -> Filling NaN values of both numerical and categorical features.
   -> Correcting skewness in order to achieve a normalized model for applying model to carry out regression.
5) For modelling used Lasso.

# FUTURE WORKS
1) Need to pre process data using other ways.
2) Performing cross validation in order to check the precision of model.
3) Using stacking models for prediction.

# FEEL FREE TO POST ANY ADVICE OR TECHNIQUE TO SOLVE FUTURE WORKS
