import pandas as pd, numpy as np

def load_data(path):
    data_dict = {}
    data_file1 = open(path, "r", encoding="utf-8")
    data_text1 = data_file1.read()[1:].split("\n")[1:-1]
    for line in data_text1:
        line = line.split("$")
        if line[0] not in data_dict:
            data_dict[line[0]] = {line[1]:line[2]}
        elif line[1] not in data_dict[line[0]]:
            data_dict[line[0]][line[1]] = line[2]
        elif isinstance(data_dict[line[0]][line[1]],list):
            data_dict[line[0]][line[1]].append(line[2])
        else:
            data_dict[line[0]][line[1]]=[data_dict[line[0]][line[1]]]
            data_dict[line[0]][line[1]].append(line[2])
    return pd.DataFrame(data_dict).T


def list_reducer(x):
    '''
    Used to reduce verbose list, such as [" ",1], [1, 1]
    '''
    if not isinstance(x,list):
        return x
    if len(x)==0:
        return np.nan
    elif len(set(x))==1:
        return x[0]
    else:    
        try:
            list(map(float,x))
        except ValueError:
            return x
        else:
            return np.mean(list(map(float,x)))



def detect_sci_notation(x):
    if not isinstance(x,str):
        return False
    if "10^" in x or "E+" in x:
        return True
    return False

def translate_sci_notation(x):
    if not isinstance(x,str):
        return x
    x=x.replace("＜","<").replace("×10","E+").replace("*10^","E+").replace("^","").replace("<","").replace("﹤","").replace("﹢","+")
    try:
        x=float(x)
    except:
        return x
    return x



def replace_num(x):
    d={"０":'0',"１":'1',"２":'2', "３":'3',"４":'4', "５":'5',"６":'6',"７":'7', "８":'8',"９":'9',"＋":"+","﹢":"+",\
      "Ⅱ":"2","Ⅰ":"1","Ⅲ":"3","Ⅳ":"4","ii°":"2","i°":"1","iii°":"3","ii°":"2","iv°":"4"}
    if not isinstance(x,str):
        return x
    for t in d.items():
        x=x.replace(*t)
    return x

def replace_word(x):
    if not isinstance(x,str):
        return x
    avg_neg=-1
    avg_pos=1
    d={"阴性":avg_neg,"-":avg_neg,"阴性（-）":avg_neg,"阴性(-)":avg_neg,"--":avg_neg,\
        "+":avg_pos,"阳性(+)":avg_pos,"阳性":avg_pos,"阳性（+）":avg_pos,"重度":avg_pos*1.5,\
        "+-":(2*avg_pos+avg_neg)/3,"极弱阳":(2*avg_pos+avg_neg)/3,"阳性(低水平)":(2*avg_pos+avg_neg)/3,"弱阳":(2*avg_pos+avg_neg)/3,}
    if x in d.keys():
        return d[x]
    if x in ["阴性","阴性（-）","-","阴性(-)","--"]:
        return 0
    elif x in ["+","阳性(+)","阳性","阳性（+）","重度"]:
        return 1
    elif x in ["+-","极弱阳","阳性(低水平)","弱阳","弱阳性","弱阳性(±)","微弱阳性"]:
        return 0.5
    elif set(x) & set("阳性+()/HP，, ")==set(x):
        return int(x.count("+"))
    elif set(x) & set("阴性-()/HP，, ")==set(x):
        return int(x.count("-"))
    elif x in ['详见报告','详见报告。','见报告','详见图文报告','见纸质报告单','详见检验单','见报告单','详见化验单','详见纸质报告','详见报告单','见图文报告']:
        return 0
    elif x in ['未生长', '未检测到突变。', '未提示', '未见', '未查', '未检测到缺失。', '未检出', '未做', '未检测到缺失']:
        return 0

    return x


def Fix(x):
    if not isinstance(x,str):
        return x
    x=x.replace(" ","")
    x=replace_num(x)
    x=replace_word(x)
    return x


def try_average(x):
    if not isinstance(x,str):
        return x
    
    if len(re.findall("\d+\.\d+",x))>0:
        ans=re.findall("\d+\.\d+",x)
        if len(ans)==0:
            return x
        elif len(ans)==1:
            return float(ans[0])
        elif len(ans)==2:
            return np.mean([float(x) for x in ans])

    elif len(re.findall("\d+",x))>0:
        ans=re.findall("\d+",x)
        if len(ans)==0:
            return np.nan
        elif len(ans)==1:
            return float(ans[0])
        elif len(ans)==2:
            return np.mean([float(x) for x in ans])
    return x