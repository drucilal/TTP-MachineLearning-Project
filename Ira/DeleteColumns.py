#import data set
iowa_train = pd.read_csv('train.csv')

#remove some columns
iowa_train_filtered = iowa_train.drop(['Alley', 'PoolQC', 'Fence', 'PoolArea', 'MiscFeature'], axis = 1)