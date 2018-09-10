
import pandas as pd 
import numpy as np
import math,pickle,numbers,os,sys

current_path=os.getcwd()
data_path="\\".join(current_path.split("\\")[:-2])+"\\data"
train=pd.read_csv(data_path+"\\meinian_round1_train_20180408.csv",encoding='gbk')
train.columns=['vid','y1','y2','y3','y4','y5']

# 删除y1,y2无标签样本
train.drop(index=train.index[train['y1'].str.contains("未查|弃查")==True],inplace=True)
train.drop(index=train.index[train['y2'].str.contains("未查|弃查")==True],inplace=True)

# 中位数替换y3中的不等号数值
range_number=list(set([s for s in train.y3 if ">" in str(s)]))
replace_range_number={}
for x in range_number:
    x_num=pd.to_numeric(x.replace(">",""),errors='raise')
    m=pd.to_numeric(train.y3[pd.to_numeric(train.y3,errors='coerce')>x_num],errors='raise').median()  #中位数
    replace_range_number[x]=m
for i,v in replace_range_number.items():
    train['y3']=train['y3'].replace(i,v)


# 处理y3的伪文本
train.y3=train.y3.astype(str).str.replace("+","")
train.y3=train.y3.astype(str).str.replace("7.75轻度乳糜","7.75")
train.y3=train.y3.astype(str).str.replace("2.2.8","2.28")


# 处理标签异常的样本
train.iloc[:,1:]=train.iloc[:,1:].apply(lambda x:pd.to_numeric(x,errors="raise"),axis=0)
train.drop(index=train.index[train['y1']==0],inplace=True)
train.drop(index=train.index[train['y2']==0],inplace=True)
train.drop(index=train.index[train['y2']>10000],inplace=True)
train.drop(index=train.index[pd.isnull(train['y3'])],inplace=True)
train.drop(index=train.index[train['y5']<0],inplace=True)

train = train.reset_index(drop=True)
train.to_pickle(data_path+"\\train.pkl")
