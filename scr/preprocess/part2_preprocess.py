import pandas as pd
import numpy as np
import math,gc,pickle,numbers,sys,os
sys.path.append(os.getcwd())
from data_helper import *

current_path=os.getcwd()
data_path="\\".join(current_path.split("\\")[:-2])+"\\data"

# -------------------Load data-------------------
data2=load_data(data_path+"\\meinian_round1_data_part2_20180408.txt")
data2.reset_index(inplace=True)
data2=data2.rename(columns={"index":"vid"})

# Reduce list to element
data2=data2.applymap(list_reducer)

# 069017
checktimes=[len(i) if isinstance(i,list) else 1 for i in data2['069017'].tolist()]
data2['069017_checktimes']=checktimes
#data2['069017_checktimes'].value_counts()
normaltimes=[i.count('正常') if isinstance(i,list) else 1 if (i=='正常'or i=='未见') else 0 for i in data2['069017'].tolist()]
data2['069017_normaltimes']=normaltimes
#data2['069017_normaltimes'].value_counts()


# -------------------Multi Strategies Preprocessing-------------------
# 这里是在notebook上，一边做EDA一边预处理，比较凌乱。
def func069017(x):
    if x=='正常'or x=='未见':
        return 0
    elif isinstance(x,list):
        temp=[e for e in x if e not in ['正常','经复查','样本经复查','']]
        return int(bool(len(temp)))
    elif pd.isnull(x) or x=='':
        return -1
    else:
        return x
data2['069017']=data2['069017'].apply(lambda x:func069017(x))
data2['069017']=[x if isinstance(x,numbers.Number) else 1 for x in data2['069017'].tolist()]

# 100010
ind=data2['100010'].apply(lambda x:x==['-', '阴性'])
if any(ind):
    data2.loc[ind,'100010']='-'

def func100010(x):
    if pd.isnull(x) or x=="未见" or x=="未做" or x==" " or x=="":
        return 0
    elif "+" in str(x) and "-" not in str(x):
        return str(x).count("+")
    elif "-" in str(x) and "+" not in str(x):
        return -str(x).count("-")
    elif "+-" in str(x):
        return 0.5
    elif x=="阴性":
        return -1
    else:
        return x
data2['100010']=data2['100010'].replace({"１＋":"+","２＋":"++","透明":""})
temp=data2['100010'].apply(lambda x:func100010(x))
data2['100010']=temp

# 2233
index=data2.index[data2['2233'].apply(lambda x:isinstance(x,list))]
data2.loc[index,'2233']='阴性'
temp=data2['2233'].tolist()

# calculate average to replace nan
pos,neg,ans=[],[],[]
for i in temp:
    if isinstance(i,str):
        if any([char.isdigit() for char in i]):
            num=i.replace(" ","").replace("阴性","").replace("阳性","").replace("+","").replace("-","")
            num=float(num)
            if "阴性" in i:
                neg.append(num)
            elif "阳性" in i:
                pos.append(num)
            ans.append(num)
        else:
            ans.append(i)
    else:
        ans.append(i)

# transform ans to numeric series
def func2233(x):
    if not isinstance(x,str):
        return x
    if "阳性" in x or x=="+":
        return np.mean(pos)
    elif "阴性" in x or x=="-" or x=='--':
        return np.mean(neg)
    elif x=="+-" or x=="可疑":
        return (sum(pos)+sum(neg))/(len(neg)+len(pos))
    else:
        return x
temp=pd.Series(ans)
temp=temp.apply(lambda x:func2233(x))
data2['2233']=temp

# 2282
ind=data2.index[data2['2282'].apply(lambda x:isinstance(x,list) and any([e for e in x if e in ['阴性','-']]))]
data2.loc[ind,'2282']='阴性'

temp=data2['2282'].copy()
temp=temp.replace({'阴性':-1,'阴性（-）':-1,'-':-1,'+':1,'阳性（+）':1,'+-':0,np.nan:0,'0.13':0})
data2['2282']=temp

# 30001
data2['30001']=data2['30001'].astype(str)
AB_ind=data2['30001'][data2['30001'].str.contains("AB")].index
A_ind=data2['30001'][data2['30001'].str.contains("A")].index.difference(AB_ind)
B_ind=data2['30001'][data2['30001'].str.contains("B")].index.difference(AB_ind)
O_ind=data2['30001'][data2['30001'].str.contains("O|0")].index.difference(AB_ind)
RHneg=data2['30001'][data2['30001'].str.contains("-|阴")].index
RHpos=data2['30001'][data2['30001'].str.contains("\+|阳")].index

data2['A']=0
data2['B']=0
data2['O']=0
data2['AB']=0
data2['RHpos']=0
data2['RHneg']=0

data2.loc[A_ind,'A']=1
data2.loc[B_ind,'B']=1
data2.loc[O_ind,'O']=1
data2.loc[AB_ind,'AB']=1
data2.loc[RHneg,'RHneg']=1
data2.loc[RHpos,'RHpos']=1

data2=data2.drop('30001',axis=1)


# 300044
data2['300044']=data2['300044'].apply(lambda x:'-' if isinstance(x,list) and any([e for e in x if any([p for p in ['-','阴'] if p in e])]) else x)

NumTemp=data2['300044'].apply(lambda x:pd.to_numeric(x,errors='coerce'))
NumTemp.loc[pd.isnull(NumTemp)]=np.mean(NumTemp.loc[~pd.isnull(NumTemp)])
SignTemp=data2['300044'].apply(lambda x:-1 if "阴" in str(x) or x=="-" else 1 if "阳" in str(x) or x=="+" else 0)
#SignTemp.value_counts()
data2['300044_num']=NumTemp
data2['300044_sign']=SignTemp
data2=data2.drop(['300044'],axis=1)

# 300073
ind=data2.index[data2['300073'].apply(lambda x: isinstance(x,list))]
data2.loc[ind,'300073']='0.84'
temp=data2['300073']
temp=pd.to_numeric(temp.replace({'<0.10':0.10,'<0.20':0.20,"<2.00":2.00,"<1.00":1.00,"< 0.10":0.10,"<1.2":1.2,"<1.20":1.20}),errors='raise')
#temp.value_counts(dropna=False)
avg=np.mean(temp.dropna())
temp=temp.replace({np.nan:avg}) # average replace nan
data2['300073']=temp

# 312
def func312(x):
    if not isinstance(x,list):
        return x
    elif len(x)==1:
        return x[0]
    elif len(x) >2:
        return x
    else:
        ans=[e for e in x if "+" not in e and "-" not in e and  "未见" not in e]
        #print(ans)
        return  ans

temp=data2['312'].apply(lambda x:func312(x))
temp=temp.apply(lambda x:pd.to_numeric(x,errors='ignore'))
temp=temp.replace({'1-5':3,'0-1':0.5,'1-3':2})
temp=temp.apply(lambda x:pd.to_numeric(x,errors='raise'))
avg=np.mean(temp.dropna())
temp=temp.replace({np.nan:avg})
# temp.value_counts(dropna=False)
data2['312']=temp

# 3194
data2['3194']=data2['3194'].apply(lambda x:'-' if isinstance(x,list) and any([e for e in x if any([p for p in ['-','阴'] if p in e])]) else x)

temp=data2['3194'].replace({"-":-1,"阴性":-1,"+-":0.5,"+":1,"++":2,"+++":3,'阳性(+)':1,"未做":np.nan,'565':np.nan})
#temp.value_counts(dropna=False)
temp=pd.to_numeric(temp,errors='raise')
avg=np.mean(temp.dropna())
temp=temp.replace({np.nan:avg})
#temp.value_counts(dropna=False)
data2['3194']=temp


# 3429
temp=data2['3429'].apply(lambda x:x if not isinstance(x,list) else "未见" if "未见" in x else x[0])
# temp.value_counts(dropna=False)

def ExtractMean(x):
    """
    deal with string like "3-5" 
    """
    temp=list(re.match(r'(\d+)[-](\d+)',x).groups())
    temp=[float(i) for i in temp]
    return np.mean(temp)

HPnum=temp.apply(lambda x: x if "HP" in str(x) and "+" not in str(x) else np.nan)
HPnum=HPnum.apply(lambda x:ExtractMean(x) if isinstance(x,str) else x)
#HPnum.value_counts(dropna=False)

pos=temp.apply(lambda x:x if "+" in str(x) or "阳" in str(x) else np.nan)
pos=pos.apply(lambda x: x.count("+") if isinstance(x,str) else x)
#pos=pos.replace({np.nan:0})

Indpos=pos[~pos.isnull()].index
IndHPnum=HPnum[~HPnum.isnull()].index
t=Indpos.tolist().copy()
t.extend(IndHPnum)

temp2=temp.copy()
temp2[t]=np.nan
def ExtractMean(x):
    try:
        temp=list(re.match(r'(\d+)[-](\d+)',x).groups())
    except (TypeError,AttributeError) as e:
        return x
    else:
        temp=[float(i) for i in temp]
        return np.mean(temp)
temp3=temp2.apply(lambda x:ExtractMean(x))
temp3=temp3.replace({"未见":0,"阴性":0,"未检出":0,"正常":0,"-":0})
temp3=temp3.apply(lambda x:pd.to_numeric(x,errors='coerce'))
avg=np.mean(temp3.dropna())
temp3=temp3.replace({np.nan:avg})

data2['3429']=temp3
data2['3429_HPnum']=HPnum
data2['3429_pos']=pos

# 3485
temp=data2['3485'].apply(lambda x:"阴性" if isinstance(x,list) and ("阴性" in x) else x)
d={"未见":-1,"阴性":-1,"未检出":-1,"-":-1,"无":-1,"少见":-1,\
   "+":1,"检出":1,"查见":1,"阳性(+)":1,"见TCT":1,"检到":1,"+/HP":1,"阳性":1,"+-":0.5,"1+":1,\
  "++":2,"2+":2,"+++":3,"结果见TCT":np.nan,"2-4":3,"见刮片":1,"酵母样细胞++":2,"1-3":2}
temp=temp.replace(d)
#temp.value_counts(dropna=False)
avg=np.mean(temp.dropna())
temp=temp.replace({np.nan:avg})
#temp.value_counts(dropna=False)
data2['3485']=temp

# 3486
data2['3486']=data2['3486'].apply(lambda x:'-' if isinstance(x,list) and any([e for e in x if any([p for p in ['-','阴'] if p in e])]) else x)
data2['3486']=data2['3486'].apply(lambda x:'未见' if isinstance(x,list) and any([e for e in x if e=="未见"]) else x)

#data2['3486'][data2['3486']==['阴性', '+']]='阴性'
#data2['3486'][data2['3486']==['阴性', '-']]='阴性'

d={"未见":-1,"阴性":-1,"未检出":-1,"-":-1,"无":-1,"少见":-1,\
   "+":1,"检出":1,"查见":1,"阳性(+)":1,"见TCT":1,"检到":1,"+/HP":1,"阳性":1,"+-":0.5,"1+":1,\
  "++":2,"2+":2,"+++":3,"结果见TCT":np.nan,"2-4":3,"见刮片":1,"酵母样细胞++":2,"1-3":2,"检出滴虫":1}
temp=data2['3486'].replace(d)
#temp.value_counts(dropna=False)
avg=np.mean(temp.dropna())
temp=temp.replace({np.nan:avg})
#temp.value_counts(dropna=False)
data2['3486']=temp

# I49002
data2=data2.drop('I49002',axis=1)

# 459271 , 21A014 , D59022 , 21A015 , K59033 ,279039，Rhneg
data2.loc[:,['459271','21A014','D59022','21A015','K59033','279039']]=data2.loc[:,['459271','21A014','D59022','21A015','K59033','279039']].applymap(lambda x:0 if pd.isnull(x) else 1)
data2=data2.drop('RHneg',axis=1)

# 269041
data2['269041']=data2['269041'].replace({"阴性":-1,"-":-1,"阳性(轻度)":1,"阳性(中度)":2,"阳性(重度)":3})

# 2231
temp=data2['2231'].apply(lambda x:np.nan if pd.isnull(x) else re.findall("\d+\.\d+",str(x)) if len(re.findall("\d+\.\d+",str(x))) else x)
def f(x):
    if not isinstance(x,list):
        return x
    elif len(x)==0:
        return np.nan
    elif len(x)==1:
        return float(x[0])
    try:
        return np.mean(float(x))
    except TypeError:
        return x
    return x
temp=temp.apply(lambda x:f(x))
t=temp[pd.to_numeric(temp,errors='coerce').notnull()].tolist()
avg_neg=np.mean([x for x in t if x>=1])
avg_pos=np.mean([x for x in t if x<1])
temp=temp.replace({"阴性":avg_neg,"-":avg_neg,"阴性（-）":avg_neg,"阴性(-)":avg_neg,"+":avg_pos,"+":avg_pos,"阳性(+)":avg_pos,"阳性":avg_pos,"+-":(2*avg_pos+avg_neg)/3})
#temp.value_counts(dropna=False)
temp=temp.replace({np.nan:avg_neg})
data2['2231']=temp

# 2230
temp=data2['2230']
def f2230(x):
    if not isinstance(x,str):
        return np.nan
    if "阴性" in x or "阴性(-)" in x or x=="-":
        return -1
    elif "阳" in x or x=="+":
        return 1
    elif x=="+-":
        return 0.5
    return -1
temp=temp.apply(lambda x:f2230(x))
temp=temp.replace({np.nan:0})
data2['2230']=temp

# 2229
temp=data2['2229'].apply(lambda x:np.nan if pd.isnull(x) else re.findall("\d+\.\d+",str(x)) if len(re.findall("\d+\.\d+",str(x))) else x)
def f(x):
    if not isinstance(x,list):
        return x
    elif len(x)==0:
        return np.nan
    elif len(x)==1:
        return float(x[0])
    try:
        return np.mean(float(x))
    except TypeError:
        return x
    return x
temp=temp.apply(lambda x:f(x))
#temp[temp.notnull()].value_counts()
t=temp[pd.to_numeric(temp,errors='coerce').notnull()].tolist()
avg_neg=np.mean([x for x in t if x<1])
avg_pos=np.mean([x for x in t if x>=1])
#avg_neg,avg_pos
temp=temp.replace({"阴性":avg_neg,"-":avg_neg,"阴性（-）":avg_neg,"阴性(-)":avg_neg,\
                   "+":avg_pos,"阳性(+)":avg_pos,"阳性":avg_pos,"阳性（+）":avg_pos,"重度":avg_pos*1.5,\
                   "+-":(2*avg_pos+avg_neg)/3,"极弱阳":(2*avg_pos+avg_neg)/3,"阳性(低水平)":(2*avg_pos+avg_neg)/3})
#temp.value_counts(dropna=False)
temp=temp.replace({np.nan:avg_neg})
#temp.value_counts(dropna=False)
data2['2229']=temp

# 2228
temp=data2['2228'].apply(lambda x:np.nan if pd.isnull(x) else re.findall("\d+\.\d+",str(x)) if len(re.findall("\d+\.\d+",str(x))) else x)
def f(x):
    if not isinstance(x,list):
        return x
    elif len(x)==0:
        return np.nan
    elif len(x)==1:
        return float(x[0])
    try:
        return np.mean(float(x))
    except TypeError:
        return x
    return x
temp=temp.apply(lambda x:f(x))
#temp[temp.notnull()].value_counts()
t=temp[pd.to_numeric(temp,errors='coerce').notnull()].tolist()
avg_neg=np.mean([x for x in t if x<1])
avg_pos=np.mean([x for x in t if x>=1])
#avg_neg,avg_pos
temp=temp.replace({"阴性":avg_neg,"-":avg_neg,"阴性（-）":avg_neg,"阴性(-)":avg_neg,"--":avg_neg,\
                   "+":avg_pos,"阳性(+)":avg_pos,"阳性":avg_pos,"阳性（+）":avg_pos,"重度":avg_pos*1.5,\
                   "+-":(2*avg_pos+avg_neg)/3,"极弱阳":(2*avg_pos+avg_neg)/3,"阳性(低水平)":(2*avg_pos+avg_neg)/3})
#temp.value_counts(dropna=False)
temp=temp.replace({np.nan:avg_neg})
#temp.value_counts(dropna=False)
data2['2228']=pd.to_numeric(temp,errors='raise')

data2=data2.apply(lambda x:pd.to_numeric(x,errors='ignore'),axis=0)

# 30007
d={"未见异常":0,"正常":0,"-":0,"阴性":0,"见TCT":0,"见刮片":0,"结果见TCT":0,"微混":0,"结果0":0,"+":2,"yellow":2,"中度":2}

def fun30007_1(x):
    if not isinstance(x,str):
        return x
    x=x.replace("°","").replace(" ","")
    for k,v in d.items():
        x=x.replace(k,str(v))
    return 4 if x=="iv" else 5 if x=='v' else 3 if (x=="iii" or x=="III") else 2 if (x=="ii"or x=="II") else 1 if (x=="i"or x=="I") else x

def fun30007_2(x):
    if not isinstance(x,str):
        return x
    return 4 if "Ⅳ" in x else 3 if "Ⅲ" in x else 2 if "Ⅱ" in x else 1 if "Ⅰ" in x else 1 if x=="i" else x

data2['30007']=data2['30007'].apply(lambda x:fun30007_2(fun30007_1(x)))
data2['30007']=pd.to_numeric(data2['30007'])

#21A059/459166癌细胞, 019019/K59034见报告, 569001少量血型
data2.drop(columns=['21A059','019019','K59034','319194','459166','569001'],inplace=True)

# 3783
def func3783(x):
    if not isinstance(x,str):
        return x
    return 0 if x=="正常" else 1
data2['3783']=data2['3783'].apply(lambda x:func3783(x))

# 3740 粪便
def func3740(x):
    if not isinstance(x,str):
        return x
    return 0 if "软" in x else 1
data2['3740']=data2['3740'].apply(lambda x:func3740(x))

# J29004
data2['J29004']=data2['J29004'].apply(lambda x:1 if x=="淡黄色" else 2 if x=="淡红色" else 0)

# 2245
data2['2245']=data2['2245'].apply(lambda x:x if not isinstance(x,str) else -1 if "-" in x or "阴" in x else x)

# 769005
data2['769005']=data2['769005'].replace(0.0,"O").apply(lambda x:x if not isinstance(x,str) else "AB" if "AB" in x else "A" if "A" in x else "B" if "B" in x else "O" if "O" in x or "0" in x else x)
AB_ind=data2['769005'][data2['769005'].str.contains('AB')==True].index
A_ind=data2['769005'][data2['769005'].str.contains("A")==True].index.difference(AB_ind)
B_ind=data2['769005'][data2['769005'].str.contains("B")==True].index.difference(AB_ind)
O_ind=data2['769005'][data2['769005'].str.contains("O")==True].index.difference(AB_ind)
data2.loc[AB_ind,'AB']=1
data2.loc[A_ind,'A']=1
data2.loc[B_ind,'B']=1
data2.loc[O_ind,'O']=1
data2['769005']=data2['769005'].apply(lambda x:0 if pd.isnull(x) else 1)

# 2284 HIV
data2['2284']=data2['2284'].apply(lambda x:0 if pd.isnull(x) else 1)

# 319103
data2['319103']=data2['319103'].replace({0.0:"O"}).apply(lambda x:x if not isinstance(x,str) else x.upper())
data2['319103']=data2['319103'].apply(lambda x:x if not isinstance(x,str) else "AB" if "AB" in x else "A" if "A" in x else "B" if "B" in x else "O" if "O" in x or "0" in x else x)
AB_ind=data2['319103'][data2['319103'].str.contains('AB')==True].index
A_ind=data2['319103'][data2['319103'].str.contains("A")==True].index.difference(AB_ind)
B_ind=data2['319103'][data2['319103'].str.contains("B")==True].index.difference(AB_ind)
O_ind=data2['319103'][data2['319103'].str.contains("O")==True].index.difference(AB_ind)
data2.loc[AB_ind,'AB']=1
data2.loc[A_ind,'A']=1
data2.loc[B_ind,'B']=1
data2.loc[O_ind,'O']=1
data2['319103']=data2['319103'].apply(lambda x:0 if pd.isnull(x) else 1)

# 809070
data2['809070']=data2['809070'].replace({0.0:"O"}).apply(lambda x:x if not isinstance(x,str) else x.upper())
data2['809070']=data2['809070'].apply(lambda x:x if not isinstance(x,str) else "AB" if "AB" in x else "A" if "A" in x else "B" if "B" in x else "O" if "O" in x or "0" in x else x)
AB_ind=data2['809070'][data2['809070'].str.contains('AB')==True].index
A_ind=data2['809070'][data2['809070'].str.contains("A")==True].index.difference(AB_ind)
B_ind=data2['809070'][data2['809070'].str.contains("B")==True].index.difference(AB_ind)
O_ind=data2['809070'][data2['809070'].str.contains("O")==True].index.difference(AB_ind)
data2.loc[AB_ind,'AB']=1
data2.loc[A_ind,'A']=1
data2.loc[B_ind,'B']=1
data2.loc[O_ind,'O']=1
data2['809070']=data2['809070'].apply(lambda x:0 if pd.isnull(x) else 1)

# 3400
data2['3400']=data2['3400'].apply(lambda x:0 if x=='透明' else 1 if x in ['微浑','微混','混浊','浑浊'] else 2)

# 3203
data2['3203']=data2['3203'].replace({"少量":2,"大量":20})

data2.drop(columns=['769006'],inplace=True)


# -------------------处理科学计数法的数据-------------------
temp=data2.apply(lambda s:any([x for x in s.tolist() if DetectSciNotation(x)==True]))
data2.loc[:,temp.index[temp]]=data2.loc[:,temp.index[temp]].applymap(lambda x:SciNotation(x))

data2=data2.apply(lambda x:pd.to_numeric(x,errors='ignore'),axis=0)


# -------------------剩余批量处理-------------------
data2.iloc[:,1:]=data2.iloc[:,1:].applymap(lambda x:Fix(x))
data2.iloc[:,1:]=data2.iloc[:,1:].applymap(try_average)

# -------------------离群点处理-------------------
data2['2278'] = data2['2278'].apply(lambda x: x if x < 10 else np.nan)
for s in ['10004', '10009', '10013', '10014', '1110', '1171', '1337', '1363', '1451', '155', '164', '179120', '1815', '1873', '192', '193', '20002', '2163', 
          '2176', '2177', '21A017', '2228', '2247', '2250', '2277', '2333', '2371', '2372', '2376', '2386', '2387', '2389', '279034', '300005', '300011', 
          '300017', '300019', '300021', '300035', '300036', '300048', '300053', '300054', '300055', '300068', '300071', '300073', '300074', '300076',
          '300078', '300086', '300093', '300112', '300113', '300120', '300121', '300122', '300123', '300124', '300126', '300127', '300130', '300134', 
          '300144', '319155', '3200', '321', '339120', '339121', '339123', '339124', '339127', '3429', '369032', '3762', '439014', '459171', '459278',
          '669002', '669003', '669006', '669021', '669025', '809025', '979028', 
          '069025', '100007', '1117', '1362', '1814', '2168', '2392', '2393', '269028', '269029', '300001', '300069', '300070', '300087', '300092', 
          '300129', '300136', '300172', '319100', '319284', '339126', '369108', '39', '4309', '459167', '459206', '459208', '4641', '669001', '669004', 
          '669005', '699009', '739005', '809010', '809022', '839018', '979013', '979024', 'X19003']:
    std = data2[s].std()
    mean = data2[s].mean()
    data2[s] = data2[s].apply(lambda x: x if x < mean + 10 * std and x > mean - 10 * std else np.nan)


data2.to_pickle(data_path+"\\data_part2.pkl")