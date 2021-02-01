# OP-Future-House-Prices

## Project Goal:
Forecast house price at least one year in the future:
  -How the price behaves quarter-to-quarter
  -How the price behaves differently in each region
  -How the price behaves differently for each type of apartment
Visualize the result:
  -Meaningful visualization of the future prices from different models
  -Meaningful visualization of the credibility of the prediction
Obtain the forecasted result given region and house type.
Goals:
  -Help OP makes budgetary and other business decision based on loans
  -Maximize future profit, minimize the risk and solve OP’s solvency position

## Dataset:
Source of data: Tilastokeskus
Data format
  -Three csv/txt files representing 3 type of apartment: one, two and three-room
  -Each file consists of 80 rows of data represent 80 regions of Finland
  -Each regions has 58 columns represent data from 2006Q1 to 2020Q2 that is treated as a time series
Problems
  -The data has multiple missing cells not only at the beginning but also in the middle
  -Reason: no apartment sold in that period of time/ too few apartment sold
Solutions
  -Remove regions with too many missing data points (70% threshold)
  -Implement sklearn IterativeImputer to fill in missing data
  -Replace outliers created by Imputer
  
## Approach:
Linear Regression:
Prophet:
Find optimal parameters and starting year for Prophet and Linear Regression
SARIMA: Grid search for optimal parameter

