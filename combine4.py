#Combine 4 - adding more features. 
#Combine 3 onwards is for testing added variables.


##IMPORT
#import the packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler

#import the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

trainprice = train.loc[:,'SalePrice']
trainprice = pd.DataFrame(trainprice)

#Changing variables to objects:
train['MSSubClass'] = train['MSSubClass'].astype(object)
train['OverallQual'] = train['OverallQual'].astype(object)
train['OverallCond'] = train['OverallCond'].astype(object)
train['ExterCond'] = train['ExterCond'].astype(object)
train['ExterQual'] = train['ExterQual'].astype(object)
train['BsmtQual'] = train['BsmtQual'].astype(object)
train['BsmtCond'] = train['BsmtCond'].astype(object)
train['GarageQual'] = train['GarageQual'].astype(object)
train['GarageCond'] = train['GarageCond'].astype(object)
train['KitchenQual'] = train['KitchenQual'].astype(object)
train['FireplaceQu'] = train['FireplaceQu'].astype(object)
test['MSSubClass'] = test['MSSubClass'].astype(object)
test['OverallQual'] = test['OverallQual'].astype(object)
test['OverallCond'] = test['OverallCond'].astype(object)
test['ExterCond'] = test['ExterCond'].astype(object)
test['ExterQual'] = test['ExterQual'].astype(object)
test['BsmtQual'] = test['BsmtQual'].astype(object)
test['BsmtCond'] = test['BsmtCond'].astype(object)
test['GarageQual'] = test['GarageQual'].astype(object)
test['GarageCond'] = test['GarageCond'].astype(object)
test['KitchenQual'] = test['KitchenQual'].astype(object)
test['FireplaceQu'] = test['FireplaceQu'].astype(object)




#Remove outlier
train = train.drop(train[(train['GrLivArea']>4000)].index)

# SalePrice log transformation
y_log = np.log1p(train['SalePrice'])



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

# Dropping the columns with so many missing values. 
test_train = test_train.drop(columns= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id'])
#not dropping poolarea since we can use that to assume that existing value means 
#there is a pool and if data is missing, it might be an indication that there is no pool. 

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

#Adding even more features 11/14 --

# Square Root Footage
# test_train['SqrSF'] = test_train ['SalePrice'] / test_train['TotalSF'] <- no proper place to add.

# GR/TotalSF
test_train['percent_grliving'] = test_train['GrLivArea'] / test_train['TotalSF']

# Bsmt
test_train['Percent_BsmtF'] = (test_train['BsmtFinSF1'] + test_train['BsmtFinSF2']) /test_train['TotalBsmtSF']
#debugging
# test_train['percent_finishbsmt1'] = test_train['percent_finishbsmt1'].astype(int)
test_train.loc[:, "Percent_BsmtF"] = test_train.loc[:, "Percent_BsmtF"].fillna("0")

#even more variables
test_train['YrRemMinBui'] = test_train['YearRemodAdd'] - test_train['YearBuilt']
test_train['YrSoldMinRem'] = test_train['YrSold'] - test_train['YearRemodAdd']

# if test_train['YrRemMinBui'] == 0:
#     test_train['isNew'] = 1
# else:
#     test_train['isNew'] = 0
    
test_train.loc[test_train['YrRemMinBui'] == 0,'isNew'] = 1
test_train.loc[test_train['YrRemMinBui'] != 0,'isNew'] = 0
test_train.loc[:, "YrSoldMinRem"] = test_train.loc[:, "YrSoldMinRem"].fillna(0)
# test_train.YrSoldMinRem = test_train.YrSoldMinRem.fillna(test_train.YrSoldMinRem.mean())
# test_train['YrSoldMinRem'].fillna((test_train['YrSoldMinRem'].mean()), inplace=True)

# test_train['p_sf'] = trainprice['SalePrice']-1/test_train['TotalSF']




#dropping the original variables used to create
test_train = test_train.drop(columns= ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'FullBath', 'BsmtHalfBath', \
                                      'HalfBath', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'])


#changing to 

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
trainX = test_train[:1456]


# #Create Dummy variable for finished bsmt
# #not distinguishing between finish quality for basement only if the basement is unfinished
test_train['BsmtFin']= (test_train['BsmtFinType1'] != 'Unf')*1

# #listing categorical values so we can create dummy columns
# ctd = test_train
# dl = []
# for i in ctd:
#     if ctd[i].dtype == 'O':
#         dl.append(i)
# print(dl)

# #Creating Dummy variables, and dropping first instances
test_train = pd.get_dummies(test_train, columns = ['MSZoning', 'Street','LotShape','LandContour','Utilities', \
                          'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', \
                          'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', \
                          'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', \
                          'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', \
                          'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', \
                          'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', \
                          'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'],drop_first = True)

## Spliting the dataset back to train and test
#final test and train dataset
final_train = test_train.iloc[:1456,:]
final_test = test_train.iloc[1456:,:]
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
# final_trainRsale.to_csv('tr1.csv')
final_test.to_csv('te4.csv')
final_trainwithYLOG.to_csv('try4.csv')