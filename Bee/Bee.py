#import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

#import dataset
iowa_train = pd.read_csv('train.csv')
iowa_test = pd.read_csv('test.csv')

#check the columns of train dataset
iowa_train.columns

#structure of train dataset
print('original rows:', iowa_train.shape[0], 'original columns:', iowa_train.shape[1])

#structure of test dataset
print('test rows:', iowa_test.shape[0], 'test columns:', iowa_test.shape[1])

#training data information
iowa_train.info()

#testing data information
iowa_test.info()

#summary on SalePrice(target variable) from train dataset
iowa_train['SalePrice'].describe()
#All prices are greater than 0.

# Sknewness and Kurtosis
print(iowa_train['SalePrice'].skew())
print(iowa_train['SalePrice'].kurt())

#histogram of SalePrice to see the distribution 
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,4))
sns.distplot(iowa_train['SalePrice'], ax = ax1)
ax1.set_ylabel('Frequency')
ax1.set_title('SalePrice Distribution')

#QQ-plot
stats.probplot(iowa_train['SalePrice'], plot=plt)
plt.show()
#this is right skewed (violating assumptions of linear regression) so we will need to normalize. 
#-> power transformation(rightskew -> power >1) or log transformation or box cox?

#types of variables 
np.unique(iowa_train.dtypes)

#Variables with float64
iowa_train.select_dtypes(include = ['float64']).dtypes

#variables with integer
iowa_train.select_dtypes(include = ['int64']).dtypes

#Variables with object
iowa_train.select_dtypes(include = ['object']).dtypes

#Numeric variables correlation
numcor = iowa_train.corr()
colormap = plt.cm.RdBu
f, ax = plt.subplots(figsize = (9,8))
sns.heatmap(numcor, ax=ax, cmap = colormap, linewidths = 0.1)

