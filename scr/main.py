import numpy as np,pandas as pd,pickle,numbers,re,sys,os

current_path=os.getcwd()
os.chdir(current_path) 

data_path="\\".join(current_path.split("\\")[:-1])+"\\data"

print("##### Executing label_helper.py #####")
os.system("python ./preprocess/label_helper.py")

print("##### Executing part1_process_1.py #####")
os.system("python ./preprocess/part1_process_1.py")

print("##### Executing part1_process_2.py #####")
os.system("python ./preprocess/part1_process_2.py")

print("##### Executing part1_process_3.py #####")
os.system("python ./preprocess/part1_process_3.py")

print("##### Executing part2_process.py #####")
os.system("python ./preprocess/part2_process.py")

print("##### Executing feature_engineer.py #####")
os.system("python ./train/feature_engineer.py")


print("##### Executing train.py #####")
os.system("python ./train/train.py")