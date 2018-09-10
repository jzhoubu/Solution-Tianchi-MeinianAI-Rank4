import pandas as pd
import numpy as np
import math,gc,pickle,numbers,sys,os
sys.path.append(os.getcwd())
from data_helper import *

current_path=os.getcwd()
data_path="\\".join(current_path.split("\\")[:-2])+"\\data"

# load data
data1=pickle.load(open(data_path+"\\data_part1_temp2","rb"))

# 遍历所有体检项目建立词袋，构造用户画像
sexlist = ['阴道', '乳腺', '子宫', '乳房', '宫颈', '宫内', '宮壁', '外阴']
sex = data1.apply(lambda x: DetectList(x.tolist(), sexlist), axis=1)
data1['sex'] = sex.astype(int)

# #  增厚
# zenghou_row = data1.apply(lambda x: DetectList(x.tolist(), ['增厚']), axis=1)
# data1['zenghou_row'] = zenghou_row.astype(int)

#  强回声
qianghuisheng_row = data1.apply(lambda x: DetectList(x.tolist(), ['强回声']), axis=1)
data1['qianghuisheng_row'] = qianghuisheng_row.astype(int)

# 脂肪肝
zhifanggan_row = data1.apply(lambda x: DetectList(x.tolist(), ['脂肪肝']), axis=1)
data1['zhifanggan_row'] = zhifanggan_row.astype(int)

# 病变
bingbian_row = data1.apply(lambda x: DetectList(x.tolist(), ['病变']), axis=1)
data1['bingbian_row'] = bingbian_row.astype(int)

# 肿大淋巴结
zhongda_lbj_row = data1.apply(lambda x: DetectList(x.tolist(), ['肿大淋巴结']), axis=1)
data1['zhongda_lbj_row'] = zhongda_lbj_row.astype(int)

# 弥漫性
mimanxing_row = data1.apply(lambda x: DetectList(x.tolist(), ['弥漫性']), axis=1)
data1['mimanxing_row'] = mimanxing_row.astype(int)

# 牙结石
yajieshi_row = data1.apply(lambda x: DetectList(x.tolist(), ['牙结石']), axis=1)
data1['yajieshi_row'] = yajieshi_row.astype(int)

# 双侧乳腺小叶增生
ruxianzs_row = data1.apply(lambda x: DetectList(x.tolist(), ['双侧乳腺小叶增生']), axis=1)
data1['ruxianzs_row'] = ruxianzs_row.astype(int)

# 囊肿
nangzhong_row = data1.apply(lambda x: DetectList(x.tolist(), ['囊肿']), axis=1)
data1['nangzhong_row'] = nangzhong_row.astype(int)

# 慢性咽炎
mxyanyan_row = data1.apply(lambda x: DetectList(x.tolist(), ['慢性咽炎']), axis=1)
data1['mxyanyan_row'] = mxyanyan_row.astype(int)

# 骨质增生
guzhizs_row = data1.apply(lambda x: DetectList(x.tolist(), ['骨质增生']), axis=1)
data1['guzhizs_row'] = guzhizs_row.astype(int)

# 高血压
gaoxueya_row = data1.apply(lambda x: DetectList(x.tolist(), ['高血压']), axis=1)
data1['gaoxueya_row'] = gaoxueya_row.astype(int)

# 龋齿
quchi_row = data1.apply(lambda x: DetectList(x.tolist(), ['龋齿']), axis=1)
data1['quchi_row'] = quchi_row.astype(int)

# 结构紊乱
jgwenluan_row = data1.apply(lambda x: DetectList(x.tolist(), ['结构紊乱']), axis=1)
data1['jgwenluan_row'] = jgwenluan_row.astype(int)

# 胸腔积液
jiye_row = data1.apply(lambda x: DetectList(x.tolist(), ['胸腔积液']), axis=1)
data1['jiye_row'] = jiye_row.astype(int)

# 窦性心动过缓
guohuang_row = data1.apply(lambda x: DetectList(x.tolist(), ['窦性心动过缓']), axis=1)
data1['guohuang_row'] = guohuang_row.astype(int)

# 胸膜增厚
xiongmozh_row = data1.apply(lambda x: DetectList(x.tolist(), ['胸膜增厚']), axis=1)
data1['xiongmozh_row'] = xiongmozh_row.astype(int)

# 心律不齐
xinlvbuqi_row = data1.apply(lambda x: DetectList(x.tolist(), ['心律不齐']), axis=1)
data1['xinlvbuqi_row'] = xinlvbuqi_row.astype(int)

# 囊壁毛糙
maozao_row = data1.apply(lambda x: DetectList(x.tolist(), ['囊壁毛糙']), axis=1)
data1['maozao_row'] = maozao_row.astype(int)

# 小叶增生
xiaoyezs_row = data1.apply(lambda x: DetectList(x.tolist(), ['小叶增生']), axis=1)
data1['xiaoyezs_row'] = xiaoyezs_row.astype(int)

# 肝囊肿
gangnangzhong_row = data1.apply(lambda x: DetectList(x.tolist(), ['肝囊肿']), axis=1)
data1['gangnangzhong_row'] = gangnangzhong_row.astype(int)

# 子宫肌瘤
zigongjiliu_row = data1.apply(lambda x: DetectList(x.tolist(), ['子宫肌瘤']), axis=1)
data1['zigongjiliu_row'] = zigongjiliu_row.astype(int)

# 白内障
baineizhang_row = data1.apply(lambda x: DetectList(x.tolist(), ['白内障']), axis=1)
data1['baineizhang_row'] = baineizhang_row.astype(int)


# 胆囊结石
dljieshi_row = data1.apply(lambda x: DetectList(x.tolist(), ['胆囊结石']), axis=1)
data1['dljieshi_row'] = dljieshi_row.astype(int)

# 左肾结石
zsjieshi_row = data1.apply(lambda x: DetectList(x.tolist(), ['左肾结石']), axis=1)
data1['zsjieshi_row'] = zsjieshi_row.astype(int)

# 前列腺增生
qlxzs_row = data1.apply(lambda x: DetectList(x.tolist(), ['前列腺增生']), axis=1)
data1['qlxzs_row'] = qlxzs_row.astype(int)

# 左肾囊肿
zsnangzhong_row = data1.apply(lambda x: DetectList(x.tolist(), ['左肾囊肿']), axis=1)
data1['zsnangzhong_row'] = zsnangzhong_row.astype(int)

# 颈椎骨质增生
jingzhuizs_row = data1.apply(lambda x: DetectList(x.tolist(), ['颈椎骨质增生']), axis=1)
data1['jingzhuizs_row'] = jingzhuizs_row.astype(int)

# 右肾囊肿
youshennz_row = data1.apply(lambda x: DetectList(x.tolist(), ['右肾囊肿']), axis=1)
data1['youshennz_row'] = youshennz_row.astype(int)

# 淋巴结肿大
linbajiezd_row = data1.apply(lambda x: DetectList(x.tolist(), ['淋巴结肿大']), axis=1)
data1['linbajiezd_row'] = linbajiezd_row.astype(int)

# 钙化灶
gaihuazao_row = data1.apply(lambda x: DetectList(x.tolist(), ['钙化灶']), axis=1)
data1['gaihuazao_row'] = gaihuazao_row.astype(int)

# 缺齿
quechi_row = data1.apply(lambda x: DetectList(x.tolist(), ['缺齿']), axis=1)
data1['quechi_row'] = quechi_row.astype(int)

# 萎缩
weisuo_row = data1.apply(lambda x: DetectList(x.tolist(), ['萎缩']), axis=1)
data1['weisuo_row'] = weisuo_row.astype(int)

# 充血
chongxue_row = data1.apply(lambda x: DetectList(x.tolist(), ['充血']), axis=1)
data1['chongxue_row'] = chongxue_row.astype(int)

# 有压痛
youyatong_row = data1.apply(lambda x: DetectList(x.tolist(), ['有压痛']), axis=1)
data1['youyatong_row'] = youyatong_row.astype(int)

# 阴道炎
yindaoyan_row = data1.apply(lambda x: DetectList(x.tolist(), ['阴道炎']), axis=1)
data1['yindaoyan_row'] = yindaoyan_row.astype(int)

# 宫颈炎症
gongjinyan_row = data1.apply(lambda x: DetectList(x.tolist(), ['宫颈炎症']), axis=1)
data1['gongjinyan_row'] = gongjinyan_row.astype(int)

# 轻度炎症
qinduyanzheng_row = data1.apply(lambda x: DetectList(x.tolist(), ['轻度炎症']), axis=1)
data1['qinduyanzheng_row'] = qinduyanzheng_row.astype(int)

# 绝经
juejing_row = data1.apply(lambda x: DetectList(x.tolist(), ['绝经']), axis=1)
data1['juejing_row'] = juejing_row.astype(int)

# 阑尾炎术后
lanweiyan_row = data1.apply(lambda x: DetectList(x.tolist(), ['阑尾炎术后']), axis=1)
data1['lanweiyan_row'] = lanweiyan_row.astype(int)

# 建议进一步检查
jinyibu_row = data1.apply(lambda x: DetectList(x.tolist(), ['建议进一步检查']), axis=1)
data1['jinyibu_row'] = jinyibu_row.astype(int)

# 血管瘤
xueguanliu_row = data1.apply(lambda x: DetectList(x.tolist(), ['血管瘤']), axis=1)
data1['xueguanliu_row'] = xueguanliu_row.astype(int)

# 牙龈炎
yayinyan_row = data1.apply(lambda x: DetectList(x.tolist(), ['牙龈炎']), axis=1)
data1['yayinyan_row'] = yayinyan_row.astype(int)

# 鼻炎
biyan_row = data1.apply(lambda x: DetectList(x.tolist(), ['鼻炎']), axis=1)
data1['biyan_row'] = biyan_row.astype(int)

# 楔状缺损
qizhuangqs_row = data1.apply(lambda x: DetectList(x.tolist(), ['楔状缺损']), axis=1)
data1['qizhuangqs_row'] = qizhuangqs_row.astype(int)

# 建议复查
fucha_row = data1.apply(lambda x: DetectList(x.tolist(), ['建议复查']), axis=1)
data1['fucha_row'] = fucha_row.astype(int)

# 随访
suifang_row = data1.apply(lambda x: DetectList(x.tolist(), ['随访']), axis=1)
data1['suifang_row'] = suifang_row.astype(int)

# 牙周炎
yazhouyan_row = data1.apply(lambda x: DetectList(x.tolist(), ['牙周炎']), axis=1)
data1['yazhouyan_row'] = yazhouyan_row.astype(int)

# 右肾结石
youshenjs_row = data1.apply(lambda x: DetectList(x.tolist(), ['右肾结石']), axis=1)
data1['youshenjs_row'] = youshenjs_row.astype(int)

# 形态失常
shichang_row = data1.apply(lambda x: DetectList(x.tolist(), ['形态失常']), axis=1)
data1['shichang_row'] = shichang_row.astype(int)

# 颈椎病
jinzhuibing_row = data1.apply(lambda x: DetectList(x.tolist(), ['颈椎病']), axis=1)
data1['jinzhuibing_row'] = jinzhuibing_row.astype(int)

# 胸膜肥厚
xiongmofh_row = data1.apply(lambda x: DetectList(x.tolist(), ['胸膜肥厚']), axis=1)
data1['xiongmofh_row'] = xiongmofh_row.astype(int)

# 前列腺增大
qianliexianzd_row = data1.apply(lambda x: DetectList(x.tolist(), ['前列腺增大']), axis=1)
data1['qianliexianzd_row'] = qianliexianzd_row.astype(int)

# 胆固醇
danguchun_row = data1.apply(lambda x: DetectList(x.tolist(), ['胆固醇']), axis=1)
data1['danguchun_row'] = danguchun_row.astype(int)

# 肾结晶
shenjiejing_row = data1.apply(lambda x: DetectList(x.tolist(), ['肾结晶']), axis=1)
data1['shenjiejing_row'] = shenjiejing_row.astype(int)

# 缺血灶
quexuezao_row = data1.apply(lambda x: DetectList(x.tolist(), ['缺血灶']), axis=1)
data1['quexuezao_row'] = quexuezao_row.astype(int)

# 血管弹性降低
tanxingjiangdi_row = data1.apply(lambda x: DetectList(x.tolist(), ['血管弹性降低']), axis=1)
data1['tanxingjiangdi_row'] = tanxingjiangdi_row.astype(int)

# 血流速度增快/ 血流速度略增快
xlsuduzengkuai_row = data1.apply(lambda x: DetectList(x.tolist(), ['血流速度增快']), axis=1)
data1['xlsuduzengkuai_row'] = xlsuduzengkuai_row.astype(int)

# 血流速度略增快
luezengkuai_row = data1.apply(lambda x: DetectList(x.tolist(), [ '血流速度略增快']), axis=1)
data1['luezengkuai_row'] = luezengkuai_row.astype(int)

# 血流速度减慢
xlsudujianman_row = data1.apply(lambda x: DetectList(x.tolist(), ['血流速度减慢']), axis=1)
data1['xlsudujianman_row'] = xlsudujianman_row.astype(int)

# 血流速度略减慢
luejianman_row = data1.apply(lambda x: DetectList(x.tolist(), ['血流速度略减慢']), axis=1)
data1['luejianman_row'] = luejianman_row.astype(int)

# 脑血管顺应性降低 / 动脉顺应性降低
shunyingxingjd_row = data1.apply(lambda x: DetectList(x.tolist(), ['脑血管顺应性降低', '动脉顺应性降低']), axis=1)
data1['shunyingxingjd_row'] = shunyingxingjd_row.astype(int)

# 血管弹性轻度减弱
xgtangxingqdjr_row = data1.apply(lambda x: DetectList(x.tolist(), ['血管弹性轻度减弱']), axis=1)
data1['xgtangxingqdjr_row'] = xgtangxingqdjr_row.astype(int)

# 血管弹性中度减弱
xgtangxingzdjr_row = data1.apply(lambda x: DetectList(x.tolist(), ['血管弹性中度减弱']), axis=1)
data1['xgtangxingzdjr_row'] = xgtangxingzdjr_row.astype(int)

# 血管弹性重度减弱
xgtangxingyzjr_row = data1.apply(lambda x: DetectList(x.tolist(), ['血管弹性重度减弱']), axis=1)
data1['xgtangxingyzjr_row'] = xgtangxingyzjr_row.astype(int)

# 0730 有无义齿
hasyichi_row = (data1['0730'] == '无')
data1['hasyichi_row'] = hasyichi_row.astype(int)

# 0972 外痔
data1['waizhi_row'] = np.nan
data1['waizhi_row'][data1['0972'].str.contains('外痔')==True] = 1
data1['waizhi_row'][data1['0972'].str.contains('外痔')==False] = 0

# 0972  内痔
data1['neizhi_row'] = np.nan
data1['neizhi_row'][data1['0972'].str.contains('内痔')==True] = 1
data1['neizhi_row'][data1['0972'].str.contains('内痔')==False] = 0

# 乳腺结节
ruxianjiejie_row = data1.apply(lambda x: DetectList(x.tolist(), ['乳腺结节']), axis=1)
data1['ruxianjiejie_row'] = ruxianjiejie_row.astype(int)

# 心动过速
xindongguosu_row = data1.apply(lambda x: DetectList(x.tolist(), ['心动过速']), axis=1)
data1['xindongguosu_row'] = xindongguosu_row.astype(int)

# 传导阻滞
chuandaozuzhi_row = data1.apply(lambda x: DetectList(x.tolist(), ['传导阻滞']), axis=1)
data1['chuandaozuzhi_row'] = chuandaozuzhi_row.astype(int)

# 心肌梗塞
gengse_row = data1.apply(lambda x: DetectList(x.tolist(), ['心肌梗塞']), axis=1)
data1['gengse_row'] = gengse_row.astype(int)

# 高电压
gaodianya_row = data1.apply(lambda x: DetectList(x.tolist(), ['高电压']), axis=1)
data1['gaodianya_row'] = gaodianya_row.astype(int)

# T波改变
tbogaibian_row = data1.apply(lambda x: DetectList(x.tolist(), ['T波改变']), axis=1)
data1['tbogaibian_row'] = tbogaibian_row.astype(int)

# 心电轴左偏
zhouzuopian_row = data1.apply(lambda x: DetectList(x.tolist(), ['心电轴左偏']), axis=1)
data1['zhouzuopian_row'] = zhouzuopian_row.astype(int)

# 心电轴右偏
zhouyoupian_row = data1.apply(lambda x: DetectList(x.tolist(), ['心电轴右偏']), axis=1)
data1['zhouyoupian_row'] = zhouyoupian_row.astype(int)

# T波低平
tbodiping_row = data1.apply(lambda x: DetectList(x.tolist(), ['T波低平']), axis=1)
data1['tbodiping_row'] = tbodiping_row.astype(int)

# 早搏
zaobo_row = data1.apply(lambda x: DetectList(x.tolist(), ['早搏']), axis=1)
data1['zaobo_row'] = zaobo_row.astype(int)

# 甲状腺结节
jiazhuangxianjj_row = data1.apply(lambda x: DetectList(x.tolist(), ['甲状腺结节']), axis=1)
data1['jiazhuangxianjj_row'] = jiazhuangxianjj_row.astype(int)

# 颈椎生理曲度变直
qudubianzhi_row = data1.apply(lambda x: DetectList(x.tolist(), ['颈椎生理曲度变直']), axis=1)
data1['qudubianzhi_row'] = qudubianzhi_row.astype(int)

# 颈椎退行性变
tuixing_row = data1.apply(lambda x: DetectList(x.tolist(), ['颈椎退行性变']), axis=1)
data1['tuixing_row'] = tuixing_row.astype(int)

# 老年
laonian_row = data1.apply(lambda x: DetectList(x.tolist(), ['老年']), axis=1)
data1['laonian_row'] = laonian_row.astype(int)

# 删除剩余的包含字符串的字段
str_series=FindStrSeries(data1.iloc[:,1:])
data1.drop(columns=str_series,inplace=True)
data1.iloc[:,1:]=data1.iloc[:,1:].apply(lambda x:pd.to_numeric(x,errors='raise'))

data1.to_pickle(data_path+"\\data_part1.pkl")
