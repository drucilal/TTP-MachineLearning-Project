#This py file is for putting everyone's work together.
#Please update on your own py file under your folder first and try it out 
#before putting it up here. ALL EDA is found here.

# Housing EDA Code




##IMPORT
#import the packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

#import the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')





##UNDERSTADING THE DATA
#check the columns of train dataset
train.columns

#structure of train dataset
print('original rows:', train.shape[0], 'original columns:', train.shape[1])

#structure of test dataset
print('test rows:', test.shape[0], 'test columns:', test.shape[1])

#training data information
train.info()

#testing data information
test.info()

#types of variables 
np.unique(train.dtypes)

#Variables with float64
train.select_dtypes(include = ['float64']).dtypes

#variables with integer
train.select_dtypes(include = ['int64']).dtypes

#Variables with object
train.select_dtypes(include = ['object']).dtypes






##Variables
#Numerical Variables
numerics = ['int64', 'float64']
numeric_train  = train.select_dtypes(include = numerics)  
numeric_train = numeric_train.drop(columns = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','GarageYrBlt', 'MoSold', 'YrSold', 'YearRemodAdd', 'OverallQual',
                                             'OverallCond', 'YearBuilt','MSSubClass'])
numeric_train.head()

# Factor Variables
factors = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
train_factors = train[factors]
train_factors.head()

# Categorical Variables: Turned into Dummies by Owner
categories = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
train_categories = train[categories]
train_categories.head()

# Categorical Variables 
categori = ['object']
catego = train.select_dtypes(include= categori)
catego.head()
categorical_train = pd.concat([catego, train_categories], axis=1, sort=False)
categorical_train.head()







##UNDERSTANDING THE SALEPRICE
#summary on SalePrice(target variable) from train dataset
train['SalePrice'].describe()
#All prices are greater than 0.

# How expensive are houses?
import matplotlib.pyplot as plt
print('The cheapest house sold for ${:,.0f} and the most expensive for ${:,.0f}'.format(
    train.SalePrice.min(), train.SalePrice.max()))
print('The average sales price is ${:,.0f}, while median is ${:,.0f}'.format(
    train.SalePrice.mean(), train.SalePrice.median()))
train.SalePrice.hist(bins=75, rwidth=.8, figsize=(14,4))
plt.title('How expensive are houses?')
plt.show()

#Remove outlier
train = train.drop(train[(train['GrLivArea']>4000)].index)

#Check if outlier was removed successfully
fig, ax = plt.subplots()
ax = sns.regplot(train['GrLivArea'], train['SalePrice'], scatter_kws={'s': 10})
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

train.shape   #4 Rows Gone from the removing the outlier


# Sales Price
print('Skew: {:.3f} | Kurtosis: {:.3f}'.format(train.SalePrice.skew(), train.SalePrice.kurtosis()))
#skew: 1.566, kurtosis: 3.885

#histogram of SalePrice to see the distribution 
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,4))
sns.distplot(train['SalePrice'], ax = ax1)
ax1.set_ylabel('Frequency')
ax1.set_title('SalePrice Distribution')
#QQ-plot
stats.probplot(train['SalePrice'], plot=plt)
plt.show()
#this is right skewed (violating assumptions of linear regression) so we will need to normalize. 
#by doing log transformation 

# SalePrice log transformation
y_log = np.log1p(train['SalePrice'])

#histogram of SalePrice to see the distribution after log transformation
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,4))
sns.distplot(y_log, ax = ax1)
ax1.set_ylabel('Frequency')
ax1.set_title('SalePrice Distribution')
#QQ-plot
stats.probplot(y_log, plot=plt)
plt.show()









##EDA
# Here is a scatter plot with dist plot for all numeric variables in the train data 
#by Sale Price
sns.jointplot(x="LotFrontage", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="LotArea", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="MasVnrArea", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="BsmtFinSF1", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="1stFlrSF", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="2ndFlrSF", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x='LowQualFinSF', y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="GrLivArea", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="BedroomAbvGr", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="KitchenAbvGr", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="TotRmsAbvGrd", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="GarageCars", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="GarageArea", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="WoodDeckSF", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="OpenPorchSF", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="EnclosedPorch", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="3SsnPorch", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="ScreenPorch", y="SalePrice", data=numeric_train, kind = 'reg')
sns.jointplot(x="MiscVal", y="SalePrice", data=numeric_train, kind = 'reg')

## Distribution Plots for Numerical Features
# Grid of distribution plots of all numerical features
f = pd.melt(numeric_train, value_vars=sorted(numeric_train))
g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')

#coorelation
#Numeric variables correlation
numcor = train.corr()
colormap = plt.cm.RdBu
f, ax = plt.subplots(figsize = (9,8))
sns.heatmap(numcor, ax=ax, cmap = colormap, linewidths = 0.1)

#Which ones are highly correlated?
s = numcor.unstack()
s[(abs(s)>0.6) & (abs(s) < 1)]

# Categorical Variables
categorical_train.columns
f = pd.melt(categorical_train, value_vars=sorted(categorical_train))
g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False)
plt.xticks(rotation='vertical')
g = g.map(sns.countplot, 'value')
[plt.setp(ax.get_xticklabels(), rotation=60) for ax in g.axes.flat]
g.fig.tight_layout()
plt.show()

# Box Plot for Categorical Features
f = pd.melt(train, id_vars=['SalePrice'], value_vars=sorted(categorical_train))
g = sns.FacetGrid(f, col='variable', col_wrap=3, sharex=False, sharey=False, size=4)
g = g.map(sns.boxplot, 'value', 'SalePrice')
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
g.fig.tight_layout()
plt.show()

# Graphing Factors
f = pd.melt(train_factors, value_vars=sorted(train_factors))
g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False)
plt.xticks(rotation='vertical')
g = g.map(sns.countplot, 'value')
[plt.setp(ax.get_xticklabels(), rotation=60) for ax in g.axes.flat]
g.fig.tight_layout()
plt.show()

# When were the houses built?
print('Oldest house built in {}. Newest house built in {}.'.format(
    train.YearBuilt.min(), train.YearBuilt.max()))
train.YearBuilt.hist(bins=14, rwidth=.9, figsize=(12,4))
plt.title('When were the houses built?')
plt.show()





#### Data Cleaning: Combination of Train and Test Data Set

##MISSING
# Missing Values Data Frame: Train 
missing = train.isna().sum()
missing = missing[missing>0]
missing_percent = missing/train.shape[0] * 100
train_missing = pd.DataFrame([missing, missing_percent], index = ['total', 'missing percent']).T
train_missing.sort_values(['missing percent'], ascending = [False])

# Missing Values Data Frame: Test
missing_test = test.isna().sum()
missing_test = missing_test[missing_test>0]
missingtest_percent = missing_test/test.shape[0] * 100
test_missing = pd.DataFrame([missing_test, missingtest_percent], index = ['total', 'missing percent']).T
test_missing.sort_values(['missing percent'], ascending = [False])

# There are some that has missing only in train dataset and only in test dataset.
# first drop the SalePrice column of train dataset and
# then we will combine two dataset and then clean it. 
trainX = train.drop('SalePrice', axis =1)     #1456 rows with 80 columns
testX = test                                  #1459 rows with 80 columns
test_train = pd.concat([trainX, testX], keys=['train', 'test'])

#Check the test_train dataset
test_train.shape
#2915 rows, 80 columns

# Dropping the columns with so many missing values. 
test_train = test_train.drop(columns= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id'])
#not dropping poolarea since we can use that to assume that existing value means 
#there is a pool and if data is missing, it might be an indication that there is no pool. 

#Check the test_train dataset
test_train.shape
#2915 rows, 75 columns


# Check the original Missing Values Data Frame: Test_Train 
missing = test_train.isna().sum()
missing = missing[missing>0]
missing_percent = missing/test_train.shape[0] * 100
test_train_missing = pd.DataFrame([missing, missing_percent], index = ['total', 'missing percent']).T
test_train_missing.sort_values(['missing percent'], ascending = [False])

##imputation
# Preprocessing: Imputation: Filling Missing Values 
test_train.loc[:, "BedroomAbvGr"] = test_train.loc[:, "BedroomAbvGr"].fillna(0)
test_train.loc[:, "BsmtQual"] = test_train.loc[:, "BsmtQual"].fillna("No")
test_train.loc[:, "BsmtCond"] = test_train.loc[:, "BsmtCond"].fillna("No")
test_train.loc[:, "BsmtExposure"] = test_train.loc[:, "BsmtExposure"].fillna("No")
test_train.loc[:, "BsmtFinType1"] = test_train.loc[:, "BsmtFinType1"].fillna("No")
test_train.loc[:, "BsmtFinType2"] = test_train.loc[:, "BsmtFinType2"].fillna("No")
test_train.loc[:, "BsmtFullBath"] = test_train.loc[:, "BsmtFullBath"].fillna(0)
test_train.loc[:, "BsmtHalfBath"] = test_train.loc[:, "BsmtHalfBath"].fillna(0)
test_train.loc[:, "BsmtUnfSF"] = test_train.loc[:, "BsmtUnfSF"].fillna(0)
test_train.loc[:, "CentralAir"] = test_train.loc[:, "CentralAir"].fillna("N")
test_train.loc[:, "Condition1"] = test_train.loc[:, "Condition1"].fillna("Norm")
test_train.loc[:, "Condition2"] = test_train.loc[:, "Condition2"].fillna("Norm")
test_train.loc[:, "EnclosedPorch"] = test_train.loc[:, "EnclosedPorch"].fillna(0)
test_train.loc[:, "ExterCond"] = test_train.loc[:, "ExterCond"].fillna("TA")
test_train.loc[:, "ExterQual"] = test_train.loc[:, "ExterQual"].fillna("TA")
test_train.loc[:, "FireplaceQu"] = test_train.loc[:, "FireplaceQu"].fillna("No")
test_train.loc[:, "Fireplaces"] = test_train.loc[:, "Fireplaces"].fillna(0)
test_train.loc[:, "Functional"] = test_train.loc[:, "Functional"].fillna("Typ")
test_train.loc[:, "GarageType"] = test_train.loc[:, "GarageType"].fillna("No")
test_train.loc[:, "GarageFinish"] = test_train.loc[:, "GarageFinish"].fillna("No")
test_train.loc[:, "GarageQual"] = test_train.loc[:, "GarageQual"].fillna("No")
test_train.loc[:, "GarageCond"] = test_train.loc[:, "GarageCond"].fillna("No")
test_train.loc[:, "GarageArea"] = test_train.loc[:, "GarageArea"].fillna(0)
test_train.loc[:, "GarageCars"] = test_train.loc[:, "GarageCars"].fillna(0)
test_train.loc[:, "HalfBath"] = test_train.loc[:, "HalfBath"].fillna(0)
test_train.loc[:, "HeatingQC"] = test_train.loc[:, "HeatingQC"].fillna("TA")
test_train.loc[:, "KitchenAbvGr"] = test_train.loc[:, "KitchenAbvGr"].fillna(0)
test_train.loc[:, "KitchenQual"] = test_train.loc[:, "KitchenQual"].fillna("TA")
test_train.loc[:, "LotFrontage"] = test_train.loc[:, "LotFrontage"].fillna(0)
test_train.loc[:, "LotShape"] = test_train.loc[:, "LotShape"].fillna("Reg")
test_train.loc[:, "MasVnrType"] = test_train.loc[:, "MasVnrType"].fillna("None")
test_train.loc[:, "MasVnrArea"] = test_train.loc[:, "MasVnrArea"].fillna(0)
test_train.loc[:, "MiscVal"] = test_train.loc[:, "MiscVal"].fillna(0)
test_train.loc[:, "OpenPorchSF"] = test_train.loc[:, "OpenPorchSF"].fillna(0)
test_train.loc[:, "PavedDrive"] = test_train.loc[:, "PavedDrive"].fillna("N")
test_train.loc[:, "SaleCondition"] = test_train.loc[:, "SaleCondition"].fillna("Normal")
test_train.loc[:, "ScreenPorch"] = test_train.loc[:, "ScreenPorch"].fillna(0)
test_train.loc[:, "TotRmsAbvGrd"] = test_train.loc[:, "TotRmsAbvGrd"].fillna(0)
test_train.loc[:, "Utilities"] = test_train.loc[:, "Utilities"].fillna("AllPub")
test_train.loc[:, "WoodDeckSF"] = test_train.loc[:, "WoodDeckSF"].fillna(0)
test_train.loc[:, "Exterior1st"] = test_train.loc[:, "Exterior1st"].fillna("No")
test_train.loc[:, "Exterior2nd"] = test_train.loc[:, "Exterior2nd"].fillna("No")
test_train.loc[:, "BsmtFinSF1"] = test_train.loc[:, "BsmtFinSF1"].fillna(0)
test_train.loc[:, "BsmtFinSF2"] = test_train.loc[:, "BsmtFinSF2"].fillna(0)
test_train.loc[:, "TotalBsmtSF"] = test_train.loc[:, "TotalBsmtSF"].fillna(0)
test_train.loc[:, "Electrical"] = test_train.loc[:, "Electrical"].fillna("Electrical")
test_train.loc[:, "SaleType"] = test_train.loc[:, "SaleType"].fillna("WD")
test_train.loc[:, "GarageYrBlt"] = test_train.loc[:, "GarageYrBlt"].fillna("0")
test_train.loc[:, "PoolArea"] = test_train.loc[:, "PoolArea"].fillna("0")
test_train.loc[:, "MSZoning"] = test_train.loc[:, "MSZoning"].fillna("RL")


# Check the Final Missing Values Data Frame: Test_Train 
missing = test_train.isna().sum()
missing = missing[missing>0]
missing_percent = missing/test_train.shape[0] * 100
test_train_missing = pd.DataFrame([missing, missing_percent], index = ['total', 'missing percent']).T
test_train_missing.sort_values(['missing percent'], ascending = [False])
#nothing missing!


#Create a variable for Total SF
#Combine all Bsmt + 1st + 2nd fl, does not distinguish between quality
test_train['TotalSF'] = test_train['TotalBsmtSF'] + test_train['1stFlrSF'] + test_train['2ndFlrSF']

#Create Variable For Total Bath
#Half Baths are multiplied by 0.5 and Full are added as a whole
test_train['TotalBath'] = test_train['BsmtFullBath'] + test_train['FullBath'] + 0.5* test_train['BsmtHalfBath'] + 0.5 * test_train['HalfBath']

#Create Variable For Total Porch SF
#We do not distinguish between the variables
test_train['TotalPorchSF'] = test_train['WoodDeckSF'] + test_train['OpenPorchSF'] + test_train['EnclosedPorch']+ test_train['3SsnPorch']+ test_train['ScreenPorch']

#find numerical variables so we can check skewness. 
a1 = test_train.dtypes[test_train.dtypes != "object"].index

skewed_features = test_train[a1].apply(lambda x: skew(x)).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_features})
skewness

#box-cox
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to BoxCox transform".format(skewness.shape[0]))
skewed_feats = skewness.index

lam = 0.15
for x in skewed_feats:
    test_train[x] = boxcox1p(test_train[x], lam)
    test_train[x] += 1

# Reassign train dataset from the transformed df
train = test_train[:1456]

# to check after the box cox
plt.figure(figsize=(20,20))
g1 = sns.jointplot(trainXLotFrontage'],y_log, s = 10)
g1.set_axis_labels('LotFrontage', 'log(SalePrice)', fontsize=12)
g2 = sns.jointplot(trainX['LotArea'],y_log, color="indianred", s = 10, xlim  = [10, 40])
g2.set_axis_labels('LotArea', 'log(SalePrice)', fontsize=12)

#Create Dummy variable for finished bsmt
#not distinguishing between finish quality for basement only if the basement is unfinished
test_train['BsmtFin']= (test_train['BsmtFinType1'] != 'Unf')*1

#listing categorical values so we can create dummy columns
ctd = test_train
dl = []
for i in ctd:
    if ctd[i].dtype == 'O':
        dl.append(i)
print(dl)

#Creating Dummy variables, and dropping first instances
test_train = pd.get_dummies(ctd, columns = ['MSZoning', 'Street','LotShape','LandContour','Utilities', \
                          'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', \
                          'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', \
                          'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', \
                          'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', \
                          'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', \
                          'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', \
                          'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'],drop_first = True)

#to check the final test_train after imputation and dummification
test_train
#2915 rows with 251 columns







## Spliting the dataset back to train and test
#final test and train dataset
final_train = df.iloc[:1456,:]
final_test = df.iloc[1456:,:]
print('final_train', final_train.shape, 'final_test', final_test.shape)
#final_train(1456,251) final_test (1459.251)

#created SalePrice df that just includes SalePrice. 
SalePrice = train.iloc[:,-1:]
SalePrice
#1456 rows and 1 column

#make final_trainRsale
final_trainRsale = final_train
final_trainRsale = final_trainRsale.reset_index()

#reset index for SalePrice
SalePrice = SalePrice.reset_index()

#put back the SalePrice to train dataset
final_trainRsale['SalePrice'] = SalePrice['SalePrice']

#get rid of level_0 and level_1 columns of final_train
del final_trainRsale['level_0']
del final_trainRsale['level_1']

#check the final_train dataset
final_trainRsale.head()

#reset index for test train
final_test = final_test.reset_index()

#get rid of level_0 and level_1 columns of final_test
del final_test['level_0']
del final_test['level_1']

#check the final_test dataset
final_test.head()

# SalePrice log series to the dataframe
y_log = y_log.to_frame()

#To check y_log
type(y_log)

#reset index for y_log
y_log = y_log.reset_index()

final_trainwithYLOG = final_train
final_trainwithYLOG = final_trainwithYLOG.reset_index()

final_trainwithYLOG['ylogSalePrice'] = y_log['SalePrice']

#get rid of level_0 and level_1 columns of final_trainwithYLOG
del final_trainwithYLOG['level_0']
del final_trainwithYLOG['level_1']

#check final_trainwithYLOG
final_trainwithYLOG.head()

#Exporting final cleaned train dataset and cleaned 
final_trainRsale.to_csv('cleanedtrain.csv')
final_test.to_csv('cleanedtest.csv')
final_trainwithYLOG.to_csv('cleanedtrainwithYlog.csv')
