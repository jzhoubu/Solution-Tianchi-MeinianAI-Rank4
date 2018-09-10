import pandas as pd
import numpy as np
import math,gc,pickle,numbers,sys,os
sys.path.append(os.getcwd())
from data_helper import *

current_path=os.getcwd()
data_path="\\".join(current_path.split("\\")[:-2])+"\\data"

# -------------------Load data-------------------
data1=load_data(data_path+"\\meinian_round1_data_part1_20180408.txt")
data1.reset_index(inplace=True)
data1=data1.rename(columns={"index":"vid"})
data1=data1.drop(columns=['1337', '1105', '39', '3705', '4503'],axis=1) #part2里有，且数据量更大
data1=data1.apply(pd.to_numeric,errors='ignore')

# -------------------Reduce list to element -------------------
data1=data1.applymap(list_reduce_part1)
# 0117，0118，0119中含有的list比例低，可以用首项取缔
data1['0117']=data1['0117'].apply(lambda x:x[0] if isinstance(x,list) else x)
data1['0118']=data1['0118'].apply(lambda x:x[0] if isinstance(x,list) else x)
data1['0119']=data1['0119'].apply(lambda x:x[0] if isinstance(x,list) else x)
# 可以判断无关的列
data1.drop(columns=['1102','1103','2501'],inplace=True)

# 批量检索关键词
data1['0101']=data1['0101'].apply(lambda x:detecter(x,keyword=['劲动脉','斑块','低回声','低回声斑块','高回声','脂肪肝'])).apply(ListReduce1)
data1['0102']=data1['0102'].apply(lambda x:detecter(x,keyword=['劲动脉','斑块','低回声','低回声斑块','高回声','脂肪肝'])).apply(ListReduce1)
data1['1308']=data1['1308'].apply(lambda x:detecter(x,keyword=['眼底','异常'])).apply(ListReduce1)
data1['A201']=data1['A201'].apply(lambda x:detecter(x,keyword=['斑块','钙化','主动脉','冠状动脉','动脉','脂肪肝'])).apply(ListReduce1)
data1['A202']=data1['A202'].apply(lambda x:detecter(x,keyword=['斑块','钙化','主动脉','冠状动脉','动脉','脂肪肝'])).apply(ListReduce1)

# 下面针对部分关键列，进行关键词检索
# 比较冗余，后面有时间的话我会重新写函数去批量处理
# 0438/0439
ind_blood_fat=\
    data1['0438'][data1['0438'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("血脂",x)))==True].index.union(\
    data1['0438'][data1['0438'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("血脂",x)))==True].index)                                                                                                                     
data1['blood_fat']=0
data1.loc[ind_blood_fat,'heart_family']=1
ind_high_blood_pressure=\
    data1['0438'][data1['0438'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("血压",x)))==True].index.union(\
    data1['0439'][data1['0439'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("血压",x)))==True].index)
data1['high_blood_pressure_family']=0
data1.loc[ind_high_blood_pressure,'high_blood_pressure_family']=1
ind_heart=\
    data1['0438'][data1['0438'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("心",x)))==True].index.union(\
    data1['0439'][data1['0439'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("心",x)))==True].index)                                                                                                                    
data1['heart_family']=0
data1.loc[ind_heart,'heart_family']=1
ind_diabetes=\
    data1['0438'][data1['0438'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("糖尿",x)))==True].index.union(\
    data1['0438'][data1['0438'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("糖尿",x)))==True].index)                                                                                                                 
data1['diabetes']=0
data1.loc[ind_diabetes,'heart_family']=1
ind_cancer=\
    data1['0438'][data1['0438'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("癌",x)))==True].index.union(\
    data1['0438'][data1['0438'].apply(lambda x:x if not isinstance(x,str) else bool(re.search("癌",x)))==True].index)                                                                                                                     
data1['cancer']=0
data1.loc[ind_cancer,'heart_family']=1
data1['0438']=data1['0438'].apply(lambda x:x if pd.isnull(x) else 0 if x=='' or re.search("[无未没]",x) else 1)
data1['0439']=data1['0439'].apply(lambda x:x if pd.isnull(x) else 0 if x=='' or re.search("[无未没]",x) else 1)

# 0912
data1['jiakang']=np.nan
data1.loc[data1['0912'][data1['0912'].str.contains("甲亢")==True].index,'jiakang']=1
data1['0912']=data1['0912'].apply(lambda x:x if pd.isnull(x) else 0 if x=='' or re.search("[无未没]",x) else 1)

#1308、1316、1330
data1['dongmaiyinghua']=np.nan
data1['eye_high_blood_pressure']=np.nan
data1.loc[data1['1308'][data1['1308'].str.contains("动脉硬化")==False].index,'dongmaiyinghua']=0
data1.loc[data1['1308'][data1['1308'].str.contains("高血压")==False].index,'eye_high_blood_pressure']=0
data1.loc[data1['1308'][data1['1308'].str.contains("动脉硬化")==True].index,'dongmaiyinghua']=1
data1.loc[data1['1308'][data1['1308'].str.contains("高血压")==True].index,'eye_high_blood_pressure']=1
data1.loc[data1['1316'][data1['1316'].str.contains("动脉硬化")==False].index,'dongmaiyinghua']=0
data1.loc[data1['1316'][data1['1316'].str.contains("高血压")==False].index,'eye_high_blood_pressure']=0
data1.loc[data1['1316'][data1['1316'].str.contains("动脉硬化")==True].index,'dongmaiyinghua']=1
data1.loc[data1['1316'][data1['1316'].str.contains("高血压")==True].index,'eye_high_blood_pressure']=1
data1.loc[data1['1330'][data1['1330'].str.contains("动脉硬化")==False].index,'dongmaiyinghua']=0
data1.loc[data1['1330'][data1['1330'].str.contains("高血压")==False].index,'eye_high_blood_pressure']=0
data1.loc[data1['1330'][data1['1330'].str.contains("动脉硬化")==True].index,'dongmaiyinghua']=1
data1.loc[data1['1330'][data1['1330'].str.contains("高血压")==True].index,'eye_high_blood_pressure']=1
data1['1308']=data1['1308'].apply(lambda x:x if pd.isnull(x) else 0)
data1['1316']=data1['1316'].apply(lambda x:x if pd.isnull(x) else 0)
data1['1330']=data1['1330'].apply(lambda x:x if pd.isnull(x) else 0)

# 1402
data1['1402_brain_dongmaiyinghua']=np.nan
data1.loc[data1['1402'][data1['1402'].str.contains("动脉硬化")==False].index,'1402_brain_dongmaiyinghua']=0
data1.loc[data1['1402'][data1['1402'].str.contains("动脉硬化")==True].index,'1402_brain_dongmaiyinghua']=1
data1['1402']=data1['1402'].apply(lambda x:x if pd.isnull(x) else 0)

# 4001
data1['4001_yinghua']=np.nan
data1.loc[data1['4001'][data1['4001'].str.contains("硬化")==False].index,'4001_yinghua']=0
data1.loc[data1['4001'][data1['4001'].str.contains("硬化")==True].index,'4001_yinghua']=1
data1['4001_gaihua']=np.nan
data1.loc[data1['4001'][data1['4001'].str.contains("硬化")==False].index,'4001_yinghua']=0
data1.loc[data1['4001'][data1['4001'].str.contains("钙化")==True].index,'4001_gaihua']=1
data1['4001']=data1['4001'].apply(lambda x:x if pd.isnull(x) else 0)

# A201
data1['A201_gaihua']=np.nan
data1.loc[data1['A201'][data1['A201'].str.contains("钙化")==False].index,'A201_gaihua']=0
data1.loc[data1['A201'][data1['A201'].str.contains("钙化")==True].index,'A201_gaihua']=1
data1['A201_zhifanggan']=np.nan
data1.loc[data1['A201'][data1['A201'].str.contains("脂肪肝")==False].index,'A201_zhifanggan']=0
data1.loc[data1['A201'][data1['A201'].str.contains("脂肪肝")==True].index,'A201_zhifanggan']=1
data1['A201']=data1['A201'].apply(lambda x:x if pd.isnull(x) else 0)

# A302
data1['A302_naogengsai']=np.nan
data1.loc[data1['A302'][data1['A302'].str.contains("脑梗")==False].index,'A302_naogengsai']=0
data1.loc[data1['A302'][data1['A302'].str.contains("脑梗")==True].index,'A302_naogengsai']=1
data1['A302']=data1['A302'].apply(lambda x:x if pd.isnull(x) else 0)

# A705
data1['A705_zhifanggan']=np.nan
data1.loc[data1['A705'][data1['A705'].str.contains("脂肪肝")==False].index,'A705_zhifanggan']=0
data1.loc[data1['A705'][data1['A705'].str.contains("脂肪肝")==True].index,'A705_zhifanggan']=1
data1['A705']=data1['A705'].apply(lambda x:x if pd.isnull(x) else 0)

# B201
data1['B201_yinghua']=np.nan
data1.loc[data1['B201'][data1['B201'].str.contains("硬化|甘油三酯")==False].index,'B201_yinghua']=0
data1.loc[data1['B201'][data1['B201'].str.contains("硬化|甘油三酯")==True].index,'B201_yinghua']=1
data1['B201_zhidanbai']=np.nan
data1.loc[data1['B201'][data1['B201'].str.contains("脂蛋白")==False].index,'B201_zhidanbai']=0
data1.loc[data1['B201'][data1['B201'].str.contains("脂蛋白")==True].index,'B201_zhidanbai']=1
data1['B201_yidao']=np.nan
data1.loc[data1['B201'][data1['B201'].str.contains("胰岛")==False].index,'B201_yidao']=0
data1.loc[data1['B201'][data1['B201'].str.contains("胰岛")==True].index,'B201_yidao']=1
data1['B201']=data1['B201'].apply(lambda x:x if pd.isnull(x) else 0)

# 1001、1002
data1['1001_quexue']=np.nan
data1.loc[data1['1001'][data1['1001'].str.contains("缺血")==False].index,'1001_quexue']=0
data1.loc[data1['1001'][data1['1001'].str.contains("缺血")==True].index,'1001_quexue']=1
data1['1001_gengsi']=np.nan
data1.loc[data1['1001'][data1['1001'].str.contains("梗死")==False].index,'1001_gengsi']=0
data1.loc[data1['1001'][data1['1001'].str.contains("梗死")==True].index,'1001_gengsi']=1
data1['1001_feida']=np.nan
data1.loc[data1['1001'][data1['1001'].str.contains("肥大")==False].index,'1001_feida']=0
data1.loc[data1['1001'][data1['1001'].str.contains("肥大")==True].index,'1001_feida']=1
data1['1001']=data1['1001'].apply(lambda x:x if pd.isnull(x) else 0)

# 0409
data1['0409_pougong'] = np.nan
data1.loc[data1.loc[data1['0409'].str.contains('剖宫')==True, '0409'].index, '0409_pougong'] = 1
data1.loc[data1.loc[data1['0409'].str.contains('剖宫')==False, '0409'].index, '0409_pougong'] = 0
data1['0409_xinlvbuqi'] = np.nan
data1.loc[data1.loc[data1['0409'].str.contains('心律不齐|早搏')==True, '0409'].index, '0409_xinlvbuqi'] = 1
data1.loc[data1.loc[data1['0409'].str.contains('心律不齐|早搏')==False, '0409'].index, '0409_xinlvbuqi'] = 0
data1['0409_high_blood_pressure'] = np.nan
data1.loc[data1.loc[data1['0409'].str.contains('血压')==True, '0409'].index, '0409_high_blood_pressure'] = 1
data1.loc[data1.loc[data1['0409'].str.contains('血压')==False, '0409'].index, '0409_high_blood_pressure'] = 0
data1['0409_xuezhi'] = np.nan
data1.loc[data1.loc[data1['0409'].str.contains('血脂')==True, '0409'].index, '0409_xuezhi'] = 1
data1.loc[data1.loc[data1['0409'].str.contains('血脂')==False, '0409'].index, '0409_xuezhi'] = 0
data1['0409_xuetang'] =np.nan
data1.loc[data1['0409'].str.contains('血糖')==True, '0409_xuetang'] = 1
data1.loc[data1['0409'].str.contains('血糖')==False, '0409_xuetang'] = 0
data1['0409_xindong'] = np.nan
data1.loc[data1['0409'].str.contains('心动过缓')==True, '0409_xindong'] = 1
data1.loc[data1['0409'].str.contains('心动过速')==False, '0409_xindong'] = 0
data1['0409_gxb'] = np.nan
data1.loc[data1['0409'].str.contains('冠心病')==True, '0409_gxb'] = 1
data1.loc[data1['0409'].str.contains('冠心病')==False, '0409_gxb'] = 0
data1['0409_zhijia'] = np.nan
data1.loc[data1['0409'].str.contains('支架')==True, '0409_zhijia'] = 1
data1.loc[data1['0409'].str.contains('支架')==False, '0409_zhijia'] = 0
data1['0409_zhifang'] = np.nan
data1.loc[data1['0409'].str.contains('脂肪')==True, '0409_zhifang'] = 1
data1.loc[data1['0409'].str.contains('脂肪')==False, '0409_zhifang'] = 0
data1['0409_tangniao'] = np.nan
data1.loc[data1['0409'].str.contains('糖尿病')==True, '0409_tangniao'] = 1
data1.loc[data1['0409'].str.contains('糖尿病')==False, '0409_tangniao'] = 0
data1['0409']=data1['0409'].apply(lambda x:x if pd.isnull(x) else 0)

# 0434
data1['0434_pougong'] = np.nan
data1.loc[data1['0434'].str.contains('剖宫')==True, '0434_pougong'] = 1
data1.loc[data1['0434'].str.contains('剖宫')==False, '0434_pougong'] = 0
data1['0434_high_blood_pressure_history'] = np.nan
data1.loc[data1['0434'].str.contains('高血压史（未治疗）')==True, '0434_high_blood_pressure_history'] = 1
data1.loc[data1['0434'].str.contains('高血压史（间断治疗）')==True, '0434_high_blood_pressure_history'] = 2
data1.loc[data1['0434'].str.contains('高血压史（治疗中）')==True, '0434_high_blood_pressure_history'] = 3
data1.loc[data1['0434'].str.contains('高血压史（中断治疗）')==True, '0434_high_blood_pressure_history'] = 4
data1['0434_high_blood_pressure'] = np.nan
data1.loc[data1['0434'].str.contains('血压')==True, '0434_high_blood_pressure'] = 1
data1.loc[data1['0434'].str.contains('血压')==False, '0434_high_blood_pressure'] = 0
data1['0434_xuezhi'] = np.nan
data1.loc[data1['0434'].str.contains('血脂')==True, '0434_xuezhi'] = 1
data1.loc[data1['0434'].str.contains('血脂')==False, '0434_xuezhi'] = 0
data1['0434_xuetang'] = np.nan
data1.loc[data1['0434'].str.contains('血糖')==True, '0434_xuetang'] = 1
data1.loc[data1['0434'].str.contains('血糖')==False, '0434_xuetang'] = 0
data1['0434_gxb'] = np.nan
data1.loc[data1['0434'].str.contains('冠心病')==True, '0434_gxb'] = 1
data1.loc[data1['0434'].str.contains('冠心病')==False, '0434_gxb'] = 0
data1['0434_zhijia'] = np.nan
data1.loc[data1['0434'].str.contains('支架')==True, '0434_zhijia'] = 1
data1.loc[data1['0434'].str.contains('支架')==False, '0434_zhijia'] = 0
data1['0434_zhifang'] = np.nan
data1.loc[data1['0434'].str.contains('脂肪')==True, '0434_zhifang'] = 1
data1.loc[data1['0434'].str.contains('脂肪')==False, '0434_zhifang'] = 0
data1['0434_tangniao'] = np.nan
data1.loc[data1['0434'].str.contains('糖尿病')==True, '0409_tangniao'] = 1
data1.loc[data1['0434'].str.contains('糖尿病')==False, '0409_tangniao'] = 0
data1['0434_cancer'] = np.nan
data1.loc[data1['0434'].str.contains('癌')==True, '0434_cancer'] = 1
data1.loc[data1['0434'].str.contains('癌')==False, '0434_cancer'] = 0
data1['0434']=data1['0434'].apply(lambda x:x if pd.isnull(x) else 0)

# 0413
data1['0413_dizhi'] = np.nan
data1.loc[data1['0413'].str.contains('低脂')==True, '0413_dizhi'] = 1
data1.loc[data1['0413'].str.contains('低脂')==False, '0413_dizhi'] = 0
data1['0413_high_blood_pressure'] = np.nan
data1.loc[data1['0413'].str.contains('血压')==True, '0413_high_blood_pressure'] = 1
data1.loc[data1['0413'].str.contains('血压')==False, '0413_high_blood_pressure'] = 0
data1['0413']=data1['0413'].apply(lambda x:x if pd.isnull(x) else 0)

# 0435
data1['0435_fubi'] = np.nan
data1.loc[data1['0435'].str.contains('腹壁')==True, '0435_fubi'] = 1
data1.loc[data1['0435'].str.contains('腹壁')==False, '0435_fubi'] = 0
data1['0435']=data1['0435'].apply(lambda x:x if pd.isnull(x) else 0)

# 0437
data1['0437_xueya'] = np.nan
data1.loc[data1['0437'].str.contains('血压')==True, '0437_xueya'] = 1
data1.loc[data1['0437'].str.contains('血压')==False, '0437_xueya'] = 0
data1['0437_xuezhi'] = np.nan
data1.loc[data1['0437'].str.contains('血脂')==True, '0437_xuezhi'] = 1
data1.loc[data1['0437'].str.contains('血脂')==False, '0437_xuezhi'] = 0
data1['0437_CM'] = np.nan
data1.loc[data1['0437'].str.contains('CM')==True, '0437_CM'] = 1
data1.loc[data1['0437'].str.contains('CM')==False, '0437_CM'] = 0
data1['0437']=data1['0437'].apply(lambda x:x if pd.isnull(x) else 0)

data1.to_pickle(data_path+"\\data_part1_temp1.pkl")





