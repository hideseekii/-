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

import pandas as pd
import numpy as np

randomstate = 50

df_1 = pd.read_csv('project_a/A_traing1/1.csv', sep=',')
df_2 = pd.read_csv('project_a/A_traing1/2.csv', sep=',')
df_3 = pd.read_csv('project_a/A_traing1/3.csv', sep=',')
df_4 = pd.read_csv('project_a/A_traing1/4.csv', sep=',')
df_5 = pd.read_csv('project_a/A_traing1/5.csv', sep=',')
#df_6 = pd.read_csv('../../TOPIC/ProjectA/A_traing2/6.csv', sep=',')

df = pd.concat((df_1,df_2,df_3,df_4,df_5),ignore_index=True)
#df = pd.concat((df_1,df_2,df_3,df_4,df_5,df_6),ignore_index=True)

print(df.describe())
x = df.iloc[:,0:5]
x[6] = df.iloc[:,5]
x[7] = df.iloc[:,6]
y1 = df.iloc[:,5]
y2 = df.iloc[:,6]
y3 = df.iloc[:,7]

print("----Standard result----")

x.iloc[:,4] = np.log10(x.iloc[:,4])
x.iloc[:,1:4] = StandardScaler().fit_transform(x.iloc[:,1:4])

print(x.describe())

data_list_X = [x.iloc[:,0:5].to_numpy()]
data_list_y1 = [y1.to_numpy()]
data_list_y2 = [y2.to_numpy()]
data_list_y3 = [y3.to_numpy()]

kf = KFold(n_splits=5, shuffle=True, random_state=randomstate)

print("----training y1----")
import xgboost
rmse = []
model1 = xgboost.XGBRegressor(objective='reg:squarederror', random_state=randomstate)
for fold, (train_index, test_index) in enumerate(kf.split(data_list_X[0])):

    X_train = data_list_X[-1][train_index]
    y_train = data_list_y1[-1][train_index]
    X_test = data_list_X[-1][test_index]
    y_test = data_list_y1[-1][test_index]
    
    # 訓練模型
    model1.fit(X_train, y_train)
    
    # 預測測試集
    y1_pred = model1.predict(X_test)
    
    # 評估模型，例如計算 RMSE
    v = np.sqrt((mean_squared_error(y_test, y1_pred)))
    rmse.append(v)

    print(f"Fold {fold+1}, Root Mean Squared Error: {v}")
    
print(f"average Root Mean Squared Error: {np.mean(rmse)}")   

#x[6] = model1.predict(data_list_X[-1])

data_list_X = [x.iloc[:,0:6].to_numpy()]

print("----training y2----")
rmse = []
model2 = xgboost.XGBRegressor(objective='reg:squarederror', random_state=randomstate)
for fold, (train_index, test_index) in enumerate(kf.split(data_list_X[0])):

    X_train = data_list_X[-1][train_index]
    y_train = data_list_y2[-1][train_index]
    X_test = data_list_X[-1][test_index]
    y_test = data_list_y2[-1][test_index]
    
    # 訓練模型
    model2.fit(X_train, y_train)
    
    # 預測測試集
    y2_pred = model2.predict(X_test)
    
    # 評估模型，例如計算 RMSE
    v = np.sqrt((mean_squared_error(y_test, y2_pred)))
    rmse.append(v)

    print(f"Fold {fold+1}, Root Mean Squared Error: {v}")
    
print(f"average Root Mean Squared Error: {np.mean(rmse)}")   

#x[7] = model2.predict(data_list_X[-1])
data_list_X = [x.iloc[:,0:7].to_numpy()]

print("----training y3----")
rmse = []
model3 = xgboost.XGBRegressor(objective='reg:squarederror', random_state=randomstate)
for fold, (train_index, test_index) in enumerate(kf.split(data_list_X[0])):

    X_train = data_list_X[-1][train_index]
    y_train = data_list_y3[-1][train_index]
    X_test = data_list_X[-1][test_index]
    y_test = data_list_y3[-1][test_index]
    
    # 訓練模型
    model3.fit(X_train, y_train)
    
    # 預測測試集
    y3_pred = model3.predict(X_test)
    
    # 評估模型，例如計算 RMSE
    v = np.sqrt((mean_squared_error(y_test, y3_pred)))
    rmse.append(v)

    print(f"Fold {fold+1}, Root Mean Squared Error: {v}")
    
print(f"average Root Mean Squared Error: {np.mean(rmse)}")  


joblib.dump(model1, 'model/A_model_y1.joblib')
joblib.dump(model2, 'model/A_model_y2.joblib')
joblib.dump(model3, 'model/A_model_y3.joblib')



def tune():
    print("----randomsearch cv----")
    # 建立參數的各自區間
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 5, 10]


    random_grid = {
                    'max_depth': max_depth, 'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                }
    random_grid

    forest2 = RandomForestRegressor(random_state=randomstate,n_estimators=1000,bootstrap=False,max_features='log2')
    rf_random = RandomizedSearchCV(estimator = forest2, param_distributions=random_grid,
                                n_iter=200, cv=5, verbose=2, random_state=randomstate, n_jobs=-1, scoring="neg_root_mean_squared_error")

    rf_random.fit(data_list_X[-1], data_list_y1[-1])
    print(rf_random.best_params_)
    
#tune()