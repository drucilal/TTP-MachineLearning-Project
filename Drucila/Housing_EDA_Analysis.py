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





