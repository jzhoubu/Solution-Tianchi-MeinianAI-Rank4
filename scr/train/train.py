import numpy as np,pandas as pd,math,gc,pickle,numbers,re,sys,os


current_path=os.getcwd()
data_path="\\".join(current_path.split("\\")[:-2])+"\\data"
train=pd.read_pickle("\\".join(data_path+"\\train.pkl")
data=pd.read_pickle(data_path+"\\FinalData.pkl")

train=train.merge(data,on='vid',how='inner')


# Train
train_x=train.iloc[:,6:].values

model1=lgb.LGBMRegressor(boosting_type='gbdt',num_leaves=50,n_estimators=3000,learning_rate=0.005,n_jobs=-1)
model1.fit(train_x,np.log1p(train['y1'].values))

model2=lgb.LGBMRegressor(boosting_type='gbdt',num_leaves=50,n_estimators=2500,learning_rate=0.005,n_jobs=-1)
model2.fit(train_x,np.log1p(train['y2'].values))

model3=lgb.LGBMRegressor(boosting_type='gbdt',num_leaves=100,n_estimators=3500,learning_rate=0.005,n_jobs=-1)
model3.fit(train_x,np.log1p(train['y3'].values))

model4=lgb.LGBMRegressor(boosting_type='gbdt',num_leaves=100,n_estimators=4200,learning_rate=0.005,n_jobs=-1)
model4.fit(train_x,np.log1p(train['y4'].values))

model5=lgb.LGBMRegressor(boosting_type='gbdt',num_leaves=100,n_estimators=3500,learning_rate=0.005,n_jobs=-1)
model5.fit(train_x,np.log1p(train['y5'].values))


# Predict
testA=pd.read_csv(data_path+"\\meinian_round1_test_a_20180409.csv",encoding='gbk')
testA=testA[['vid']]
testA=testA.merge(data,on='vid',how='inner')

test_x=testA.iloc[:,1:].values
y1=np.exp(model1.predict(test_x))-1
y2=np.exp(model2.predict(test_x))-1
y3=np.exp(model3.predict(test_x))-1
y4=np.exp(model4.predict(test_x))-1
y5=np.exp(model5.predict(test_x))-1

output=pd.DataFrame({'vid':testA.vid,'y1':y1,'y2':y2,'y3':y3,'y4':y4,'y5':y5})
output.to_csv("\\".join(os.getcwd().split("\\")[:-1]+["submit"])+"\\test_a.csv")


