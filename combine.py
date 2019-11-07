#This py file is for putting everyone's work together.
#Please update on your own py file under your folder first and try it out 
#before putting it up here. 

# Drucila LeFevre : 11/06/2019

# Housing EDA Code

import pandas as pd 
import seaborn as sns
import math

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

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

train = train.drop(columns= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'PoolArea', 'Id'])


#Numerical Variables
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
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


# Sales Price

print('Skew: {:.3f} | Kurtosis: {:.3f}'.format(numeric_train.SalePrice.skew(), numeric_train.SalePrice.kurtosis()))


# Looking at Target Variable: Sale Price
plt.figure(figsize=(10,6))
sns.distplot(numeric_train.SalePrice)
plt.show()

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



# Categorical Variables
categorical_train.columns
f = pd.melt(categorical_train, value_vars=sorted(categorical_train))
g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False)
plt.xticks(rotation='vertical')
g = g.map(sns.countplot, 'value')
[plt.setp(ax.get_xticklabels(), rotation=60) for ax in g.axes.flat]
g.fig.tight_layout()
plt.show()



# How expensive are houses?
import matplotlib.pyplot as plt
print('The cheapest house sold for ${:,.0f} and the most expensive for ${:,.0f}'.format(
    train.SalePrice.min(), train.SalePrice.max()))
print('The average sales price is ${:,.0f}, while median is ${:,.0f}'.format(
    train.SalePrice.mean(), train.SalePrice.median()))
train.SalePrice.hist(bins=75, rwidth=.8, figsize=(14,4))
plt.title('How expensive are houses?')
plt.show()

# When were the houses built?
print('Oldest house built in {}. Newest house built in {}.'.format(
    train.YearBuilt.min(), train.YearBuilt.max()))
train.YearBuilt.hist(bins=14, rwidth=.9, figsize=(12,4))
plt.title('When were the houses built?')
plt.show()



# Distribution Plots for Numerical Features
# Grid of distribution plots of all numerical features
f = pd.melt(numeric_train, value_vars=sorted(numeric_train))
g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')


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

# Data Cleaning: Combination of Train and Test Data Set


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








