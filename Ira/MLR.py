##IMPORT
#import the packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import itertools
ols = linear_model.LinearRegression()

#import the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
cleanedtrain = pd.read_csv('cleanedtrain.csv')
ctd = cleanedtrain

#Create DataFrame with all the independant variables and betas
temp_df = ctd.drop(['SalePrice'], axis=1)

#naming for bruteforce later on
temp = temp_df.iloc[:,1:]
features = temp.copy()
price   = ctd['SalePrice']

#Brute force feature selection
for i in range(0,500):
   combo = list(itertools.combinations(features.columns,5))[i]
   combo_list = list(combo)
   lm.fit(features[combo_list], price)
   scores[lm.score(features[combo_list],price)] = combo_list
   print("working...")
    
# sorted(scores.items(), key=lambda t:t[1], reverse=True)[:10]
for key in sorted(scores):
   print(key, scores[key])

