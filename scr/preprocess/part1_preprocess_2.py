import pandas as pd
import numpy as np
import math,gc,pickle,numbers,sys,os
sys.path.append(os.getcwd())
from data_helper import *

current_path=os.getcwd()
data_path="\\".join(current_path.split("\\")[:-2])+"\\data"


data1=pickle.load(open(data_path+"\\data_part1_temp1.pkl","rb"))


# -----------批量处理一些能转化为哑变量的字段-----------
for i in range(219, 222):
    index = '0' + str(i)
    data1[index] = data1[index].apply(lambda x: bool2num(x))
    
for i in range(226, 231):
    index = '0' + str(i)
    data1[index] = data1[index].apply(lambda x: bool2num(x))
    
for i in range(443, 456):
    index = '0' + str(i)
    data1[index] = data1[index].apply(lambda x: bool2num(x))
    
for i in range(403, 413):
    index = 'A' + str(i)
    data1[index] = data1[index].apply(lambda x: bool2num(x))
    
for i in range(414, 426):
    index = 'A' + str(i)
    data1[index] = data1[index].apply(lambda x: bool2num(x))

data1['0224'] = data1['0224'].apply(lambda x: bool2num(x))
data1['0236'] = data1['0236'].apply(lambda x: bool2num(x))
data1['0403'] = data1['0403'].apply(lambda x: bool2num(x))
data1['0414'] = data1['0414'].apply(lambda x: bool2num(x))
data1['0415'] = data1['0415'].apply(lambda x: bool2num(x))
data1['0428'] = data1['0428'].apply(lambda x: bool2num(x))
data1['0551'] = data1['0551'].apply(lambda x: bool2num(x))
data1['0735'] = data1['0735'].apply(lambda x: bool2num(x))
data1['0977'] = data1['0977'].apply(lambda x: bool2num(x))
data1['0980'] = data1['0980'].apply(lambda x: bool2num(x))
data1['0982'] = data1['0982'].apply(lambda x: bool2num(x))
data1['0988'] = data1['0988'].apply(lambda x: bool2num(x))
data1['1333'] = data1['1333'].apply(lambda x: bool2num(x))
data1['1340'] = data1['1340'].apply(lambda x: bool2num(x))
data1['214'] = data1['214'].apply(lambda x: bool2num(x))
data1['3704'] = data1['3704'].apply(lambda x: bool2num(x))


def bool2num_1(x):
    if pd.isnull(x) or x == "" or x == " ":
        return np.nan
    if isinstance(x, str):
#         if re.search(r"空虚", x):
#             return x
        if re.search(r"未查|弃查|未要求|不能检测", x): 
            return np.nan
        elif re.search(r"未见|正常|无|不|未发现", x):
            return 0
        else:
            return np.nan
        
data1['0214'] = data1['0214'].apply(lambda x: bool2num_1(x))
data1['0218'] = data1['0218'].apply(lambda x: bool2num_1(x))
data1['0733'] = data1['0733'].apply(lambda x: bool2num_1(x))
data1['0979'] = data1['0979'].apply(lambda x: bool2num_1(x))
data1['0986'] = data1['0986'].apply(lambda x: bool2num_1(x))
data1['0990'] = data1['0990'].apply(lambda x: bool2num_1(x))

# 0548主要数据为“适中”“否”“是”“量少”“多，黄”“量多”“多，豆渣样”（完全不懂是什么），以0表示“否”“少”，1表示“适中”，2表示“是”“多”
def func0548(x):
    if pd.isnull(x) or x == "" or x == " ":
        return np.nan
    
    if re.search(r"适中", x):
        return 1
    elif re.search(r"少|否", x):
        return 0
    elif re.search(r"多|是", x):
        return 2
    else:
        return x
    
data1['0548'] = data1['0548'].apply(lambda x: func0548(x))

# 3303
def func3303(x):
    if pd.isnull(x):
        return x
    if x == "阴性":
        return 0
    elif x == "阳性":
        return 1
    else:
        return 0.5
    
data1['3303'] = data1['3303'].apply(lambda x: func3303(x))

# 0425
def func0425(x):
    if pd.isnull(x) or x == "" or x == " ":
        return np.nan
    
    if isinstance(x, str):
        if x == "正常":
            return 0
        elif re.search(r"异常|清",  x):
            return 0
        elif re.search(r"粗糙|缓慢", x):
            return -1
        elif x == "急促":
            return 1
        elif re.search(r"\d+", x):
            num = int(re.findall(r"\d+", x)[0])
            return num

data1['0425'] = data1['0425'].apply(lambda x: func0425(x))


# -----------批量处理一些字符型的伪文本-----------
for i in range(105, 110):
    index = '0' + str(i)
    data1[index] = data1[index].apply(lambda x: str2num(x))
    
for i in range(111, 113):
    index = '0' + str(i)
    data1[index] = data1[index].apply(lambda x: str2num(x))

for i in range(2403, 2418):
    index = str(i)
    data1[index] = data1[index].apply(lambda x: str2num(x))
    
data1['2420'] = data1['2420'].apply(lambda x: str2num(x))
data1['2422'] = data1['2422'].apply(lambda x: str2num(x)) 
data1['2423'] = data1['2423'].apply(lambda x: str2num(x))
data1['2426'] = data1['2426'].apply(lambda x: str2num(x))
data1['2427'] = data1['2427'].apply(lambda x: str2num(x))
data1['2428'] = data1['2428'].apply(lambda x: str2num(x))
data1['3802'] = data1['3802'].apply(lambda x: str2num(x))
data1['3804'] = data1['3804'].apply(lambda x: str2num(x))
data1['3805'] = data1['3805'].apply(lambda x: str2num(x))
data1['3806'] = data1['3806'].apply(lambda x: str2num(x))
data1['3807'] = data1['3807'].apply(lambda x: str2num(x))
data1['3810'] = data1['3810'].apply(lambda x: str2num(x))
data1['3811'] = data1['3811'].apply(lambda x: str2num(x))
data1['A701'] = data1['A701'].apply(lambda x: str2num(x))
data1['A703'] = data1['A703'].apply(lambda x: str2num(x))


def str2num_1(x):
    if pd.isnull(x) or x == "" or x == " ":
        return np.nan
    if isinstance(x, str):
        if re.search(r"未查|弃查|未要求|不能检测", x): 
            return np.nan
        elif re.search(r', ', x):
            return 1
        elif re.search(r'正常|未见|无', x):
            return 0
        elif re.search(r'\d', x):
            return float(x)
        else:
            return 1
        
data1['0104'] = data1['0104'].apply(lambda x: str2num_1(x))
data1['0442'] = data1['0442'].apply(lambda x: str2num_1(x)) 
data1['3801'] = data1['3801'].apply(lambda x: str2num_1(x))
# 以上三个都只有“正常”或者“无”，处理成0

#以下几个都是眼科相关，“指数”“失明”“光感”“手动”等全部处理成1；另1321、1322都有“1.0，0.7”这样格式的数据，这里也处理成1
data1['1321'] = data1['1321'].apply(lambda x: str2num_1(x))
data1['1322'] = data1['1322'].apply(lambda x: str2num_1(x))
data1['1325'] = data1['1325'].apply(lambda x: str2num_1(x))
data1['1326'] = data1['1326'].apply(lambda x: str2num_1(x))
data1['2424'] = data1['2424'].apply(lambda x: str2num_1(x))
data1['2425'] = data1['2425'].apply(lambda x: str2num_1(x))






# ------------特别处理-------------
def str2bool0424(x):
    if pd.isnull(x) or x == '' or x == ' ':
        return np.nan
    
    x = x.replace(" ", "")
    if re.search("次|未查", x): 
        return np.nan
    elif re.search(r"未见|正常", x):
        return 0
    elif re.search(r"过缓|<\d+", x):
        return -1
    elif re.search(r"过速|>\d+", x):
        return 1
    elif re.search(r"\d+", x):
        if re.search(r'(\d+)--(\d+)', x):
            num = ExtractMean(x)
        else:
            num = float(re.findall(r"(?<!<)\d+", x)[0])

        if num < 60:
            return -1
        elif num > 100:
            return 1
        else:
            return 0
    else:
        return x

def str2num0424(x):
    if pd.isnull(x) or x == '' or x == ' ':
        return np.nan
    
    x = x.replace(" ", "")
    if re.search("次|未查", x):
        return np.nan
    elif re.search(r"未见|正常", x):
        return float(80.0)
    elif re.search(r"\d+", x):
        if re.search(r'(\d+)--(\d+)', x):
            num = ExtractMean(x)
        else:
            num = float(re.findall(r"(?<!<)\d+", x)[0])
        return num
    else:
        return x

def ReplaceWithSlowFast(x, slow, fast):
    if isinstance(x, str):
        if re.search(r"过缓", x):
            return slow
        elif re.search(r"过速", x):
            return fast
    else:
        return x
        
data1['0424_bool'] = data1['0424'].apply(lambda x: str2bool0424(x))
data1['0424'] = data1['0424'].apply(lambda x: str2num0424(x))

slow = float(int(np.mean([float(x) for x in data1['0424'].tolist() if isinstance(x, float) and float(x) < 60])))
fast = float(int(np.mean([float(x) for x in data1['0424'].tolist() if isinstance(x, float) and float(x) > 100])))
data1['0424'] = data1['0424'].apply(lambda x: ReplaceWithSlowFast(x, slow, fast))

def str2bool1319_1320(x):
    if pd.isnull(x) or x == '' or x == ' ':
        return np.nan
    
    x = x.replace(" ", "")
    if re.search("未查|弃查|无法|未要求", x): 
        return np.nan
    elif re.search(r"1614", x):
        return 0
    elif re.search(r"未见|正常", x) and not re.search(r'\d', x):
        return 0
    elif re.search(r"高", x):
        return -1
    elif re.search(r"低", x):
        return 1
    elif re.search(r"\d+", x):
        num = float(re.findall(r"\d+\.?\d?", x)[0])

        if num < 10:
            return -1
        elif num > 21:
            return 1
        else:
            return 0
    else:
        return x

def str2num1319_1320(x):
    if pd.isnull(x) or x == '' or x == ' ':
        return np.nan
    
    x = x.replace(" ", "")
    if re.search("未查|弃查|无法|未要求", x): 
        return np.nan
    elif re.search(r"未见|正常", x) and not re.search(r'\d', x):
        return float(15.5)
    elif re.search(r"1614", x):
        return float(16.14)
    elif re.search(r"\d+", x):
        num = float(re.findall(r"\d+\.?\d?", x)[0])
        return num
    else:
        return x
    
def ReplaceWithLowHigh(x, low, high):
    if isinstance(x, str):
        if re.search(r"低", x):
            return low
        elif re.search(r"高", x):
            return high
    else:
        return x
        
data1['1319_bool'] = data1['1319'].apply(lambda x: str2bool1319_1320(x))
data1['1319'] = data1['1319'].apply(lambda x: str2num1319_1320(x))

low_1319 = float(int(np.mean([float(x) for x in data1['1319'].tolist() if isinstance(x, float) and float(x) < 10])))
high_1319 = float(int(np.mean([float(x) for x in data1['1319'].tolist() if isinstance(x, float) and float(x) > 21])))
data1['1319'] = data1['1319'].apply(lambda x: ReplaceWithLowHigh(x, low_1319, high_1319))
data1['1319'].value_counts()

data1['1320_bool'] = data1['1320'].apply(lambda x: str2bool1319_1320(x))
data1['1320'] = data1['1320'].apply(lambda x: str2num1319_1320(x))

low_1320 = float(int(np.mean([float(x) for x in data1['1320'].tolist() if isinstance(x, float) and float(x) < 10])))
high_1320 = float(int(np.mean([float(x) for x in data1['1320'].tolist() if isinstance(x, float) and float(x) > 21])))
data1['1320'] = data1['1320'].apply(lambda x: ReplaceWithLowHigh(x, low_1320, high_1320))


def percent2num(x):
    if pd.isnull(x) or x == '' or x == ' ':
        return np.nan
    if isinstance(x, str):
        if re.search(r"未查|弃查", x):
            return np.nan
        if re.search(r"%", x):
            num = float(re.findall(r"\d+\.?\d?", x)[0]) / 100
            return num
        else:
            if re.search(r"\d", x):
                return float(x)
            else:
                return np.nan
    else:
        return x

data1['A702'] = data1['A702'].apply(lambda x: percent2num(x))
data1['A704'] = data1['A704'].apply(lambda x: percent2num(x))

def delpercent(x):
    if pd.isnull(x) or x == '' or x == ' ':
        return np.nan
    if isinstance(x, str):
        if re.search(r"未查|弃查", x):
            return np.nan
        if re.search(r"%", x):
            num = float(re.findall(r"\d+\.?\d?", x)[0]) / 100
            return num
        else:
            if re.search(r"\d", x):
                return float(x) / 100
            else:
                return x

data1['2421'] = data1['2421'].apply(lambda x: delpercent(x))
data1['3803'] = data1['3803'].apply(lambda x: delpercent(x))

def str2bool2413(x):
    if pd.isnull(x) or x == "" or x == " ":
        return np.nan
    
    for t in [int, float]:
        if isinstance(x, t):
            if x <= 75:
                return 1
            else:
                return 0

    if isinstance(x, str):
        if x == "未查":
            return np.nan
        num = float(re.findall(r"\d+", x)[0])
        if num <= 75:
            return 1
        else:
            return 0
        
def str2num2413(x):
    if pd.isnull(x) or x == "" or x == " ":
        return np.nan
    
    if isinstance(x, str):
        if re.search(r"未查", x):
            return np.nan
        num = float(re.findall(r"\d+", x)[0])
        return num
    else:
        return x

data1['2413_bool'] = data1['2413'].apply(lambda x: str2bool2413(x))
data1['2413'] = data1['2413'].apply(lambda x: str2num2413(x))

def countplus(x):
    if "双" in x:
        return x.count("＋") * 2
    else:
        return x.count("＋")
    
def teeth(x):
    if pd.isnull(x) or x == "" or x == " ":
        return np.nan
    
    if isinstance(x, str):
        if x == "无" or x == "未见明显异常":
            return 0
        elif x == "有":
            return 1
        elif re.search(r"＋", x):
            if re.search(r",", x):
                tmp_list = x.strip().split(",")
                counts = 0
                for tmp in tmp_list:
                    counts += countplus(tmp)
            else:
                counts = countplus(x)
            return counts
        elif re.search(r"松动", x) and not re.search(r"\+", x):
            counts = 1
            if re.search(r"一度", x):
                counts = counts
            elif re.search(r"二度", x):
                counts += 1
            elif re.search(r"三度", x):
                counts += 2
            return counts
        else:
            return len(x.split(" "))

data1['0715'] = data1['0715'].apply(lambda x: teeth(x))

def weights4bool(x):
    if pd.isnull(x) or x == "" or x == " ":
        return np.nan
    
    num = re.findall(r'\d+\.?\d?', x)
    if float(num[0]) < float(num[1]):
        return -1
    elif float(num[0]) > float(num[2]):
        return 1
    else:
        return 0
    
def weights4num(x):
    if pd.isnull(x) or x == "" or x == " ":
        return np.nan
    return float(re.findall(r'\d+\.?\d?', x)[0])

data1['2429_bool'] = data1['2429'].apply(lambda x: weights4bool(x))
data1['2429'] = data1['2429'].apply(lambda x: weights4num(x))

data1['2430_bool'] = data1['2430'].apply(lambda x: weights4bool(x))
data1['2430'] = data1['2430'].apply(lambda x: weights4num(x))

data1['2431_bool'] = data1['2431'].apply(lambda x: weights4bool(x))
data1['2431'] = data1['2431'].apply(lambda x: weights4num(x))
data1['2432'] = data1['2432'].apply(lambda x: weights4num(x))


data1.to_pickle(data_path+"\\data_part1_temp2.pkl")
