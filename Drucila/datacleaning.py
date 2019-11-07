
# Looking at Missing Values 

iowa_train = pd.read_csv('train.csv')
iowa_test = pd.read_csv('test.csv')

# Missing Values Data Frame : Train DataSet
missing = iowa_train.isna().sum()
missing = missing[missing>0]
missing_percent = missing/iowa_train.shape[0] * 100
train_missing = pd.DataFrame([missing, missing_percent], index = ['total', 'missing percent']).T
train_missing.sort_values(['missing percent'], ascending = [False])

# Missing Values Date Frame : Test Dataset
missing_test = iowa_test.isna().sum()
missing_test = missing_test[missing_test>0]
missingtest_percent = missing_test/iowa_test.shape[0] * 100
test_missing = pd.DataFrame([missing_test, missingtest_percent], index = ['total', 'missing percent']).T
test_missing.sort_values(['missing percent'], ascending = [False])



# Data Cleaning: Combined Data Set (Test and Train)

import pandas as pd 

# Importing Combined DataSet (Test and Train)
train = pd.read_csv('test_train.csv')

# Preprocessing: Imputation: Filling Missing Values 
train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")
train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)
train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)
train.loc[:, "Exterior1st"] = train.loc[:, "Exterior1st"].fillna("No")
train.loc[:, "Exterior2nd"] = train.loc[:, "Exterior2nd"].fillna("No")
train.loc[:, "BsmtFinSF1"] = train.loc[:, "BsmtFinSF1"].fillna(0)
train.loc[:, "BsmtFinSF2"] = train.loc[:, "BsmtFinSF2"].fillna(0)
train.loc[:, "TotalBsmtSF"] = train.loc[:, "TotalBsmtSF"].fillna(0)
train.loc[:, "Electrical"] = train.loc[:, "Electrical"].fillna("Electrical")
train.loc[:, "SaleType"] = train.loc[:, "SaleType"].fillna("WD")
train.loc[:, "GarageYrBlt"] = train.loc[:, "GarageYrBlt"].fillna("None")


# Ms Zoning and Garage Yr Blt Still * Pending* 

# Lookig at the frequency of garage year built
temp = train['GarageYrBlt'].value_counts().reset_index(name='GarageYrBlt')

# Looking at Scatterplot
sns.scatterplot(x = 'GarageCars',  y = 'GarageArea', data = train)






