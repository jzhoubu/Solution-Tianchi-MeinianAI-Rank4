import pandas as pd
import numpy as np
import math,gc,pickle,numbers,sys,os
sys.path.append(os.getcwd())
from data_helper import *

current_path=os.getcwd()
data_path="\\".join(current_path.split("\\")[:-2])+"\\data\\meinian_round1_data_part1_20180408.txt"

# -------------------Load data-------------------
data1=load_data(data_path)
data1.reset_index(inplace=True)
data1=data1.rename(columns={"index":"vid"})
data1=data1.drop(columns=['1337', '1105', '39', '3705', '4503'],axis=1) #part2里有，且数据量更大

