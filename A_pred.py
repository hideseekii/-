from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib

import xgboost
import pandas as pd
import numpy as np

randomstate = 50

df = pd.read_csv('../../TOPIC/ProjectA/A_traing1/5.csv', sep=',')
#df = pd.read_csv('../../TOPIC/ProjectA/A_testing/test.csv', sep=',')



print(df.describe())
x = df.iloc[:,0:5]

y1 = df.iloc[:,5]
y2 = df.iloc[:,6]
y3 = df.iloc[:,7]

print("----Standard result----")

x.iloc[:,4] = np.log10(x.iloc[:,4])
x.iloc[:,1:4] = StandardScaler().fit_transform(x.iloc[:,1:4])

print(x.describe())

data_list_X = [x.iloc[:,0:5].to_numpy()]



#####y1
print("----pred y1----")
model1 = joblib.load('model/A_model_y1.joblib')
X = data_list_X[-1]

# 預測測試集
y1_pred = model1.predict(X)
x['y1'] = y1_pred 
print(x.describe())   



#####y2
data_list_X = [x.iloc[:,0:6].to_numpy()]
print("----pred y2----")
model2 = joblib.load('model/A_model_y2.joblib')
X = data_list_X[-1]

# 預測測試集
y2_pred = model2.predict(X)
x['y2'] = y2_pred 
print(x.describe())  



#####y3
data_list_X = [x.iloc[:,0:7].to_numpy()]
print("----pred y3----")
model3 = joblib.load('model/A_model_y3.joblib')
X = data_list_X[-1]

# 預測測試集
y3_pred = model3.predict(X)
x['y3'] = y3_pred 
print(x.describe())  

output = x[['id','y1','y2','y3']]
print(output)
output.to_csv('submit/TOPIC/ProjectA/112060_projectA_ans.csv',index=False)