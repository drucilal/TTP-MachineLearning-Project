import pandas as pd
import seaborn as sns
from scipy import stats

iowa_train = pd.read_csv('train.csv')
iowa_train.head()


iowa_train['SalePrice'].describe()
# ALl prices are greater than zero 

# Looking at Skewness and Kurtosis 
sns.distplot(iowa_train['SalePrice'])


# Sknewness and Kurtosis
print(iowa_train['SalePrice'].skew())
print(iowa_train['SalePrice'].kurt())



# Visualzing the Numeric Variables Through ScatterPlots

# Extracting Numeric Variables from Data Frame 
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_iowatrain = iowa_train.select_dtypes(include = numerics)  
num_iowatrain.head()

num_iowatrain.columns

Gr = sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = num_iowatrain)
Lot = sns.scatterplot(x = 'LotArea', y = 'SalePrice', data = num_iowatrain)
Mas = sns.scatterplot(x = 'MasVnrArea', y = 'SalePrice', data = num_iowatrain
