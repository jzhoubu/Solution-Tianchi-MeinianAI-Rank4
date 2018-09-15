
import numpy as np,pandas as pd,math,gc,pickle,numbers,re,sys,os
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer,mean_squared_error


print("##### Loading Data #####")
current_path=os.getcwd()
data_path="\\".join(current_path.split("\\")[:-2])+"\\data"

train=pd.read_pickle(data_path+"\\train.pkl")
data1=pd.read_pickle(data_path+"\\data_part1.pkl")  
data2=pd.read_pickle(data_path+"\\data_part1.pkl") 
NLP=pd.read_pickle(data_path+"\\NLP-features.pkl")  # figure out this name you save

print("##### Merging Data #####")
data=data2.merge(data1,on='vid',how='inner')


print("##### Loading feature importance #####")
Map=pd.read_pickle("\\".join(os.getcwd().split("\\")[:-1]+["data"])+"\\feat-importance.pkl")
L=Map.index[:50].tolist()
L2=[x for x in L if len(x)==2]
L3=[x for x in L if len(x)==3]
L4=[x for x in L if len(x)==4]
L5=[x for x in L if len(x)==5]
L6=[x for x in L if len(x)==6]


print("##### Generating features with simple operation #####")

for i in range(2,7):
    L=eval("L"+str(i))
    while(len(L)-1):
        token=L.pop()
        for left in L:
            data[token+"_plus_"+left]=data[token]+data[left]
        
            data[token+"_mins_"+left]=data[token]-data[left]
        
            data[token+"_multiply_"+left]=data[token].multiply(data[left])
            
            data[token+"_div_"+left]=data[token].div(data[left])

data.shape

print("##### Droping features #####")

DropList=Map.index[Map['sum']==0]
data.drop(columns=DropList,inplace=True)
data.shape


print("##### Saving Data #####")
data=data.merge(NLP,on='vid',how='inner')
data.shape



data.to_pickle(data_path+"\\FinalData.pkl")

