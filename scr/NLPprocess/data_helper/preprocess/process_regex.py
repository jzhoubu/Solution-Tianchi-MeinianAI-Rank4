# -*- coding:utf-8 -*-

def sort_vocab_by_length(vocab):
    return sorted(vocab, key=lambda x:len(x), reverse=True)

keyword_0102 = ["肝", "胆", "胰", "脾", "左肾", "右肾", "甲状腺", "前列腺"]
keyword_0102 = sort_vocab_by_length(keyword_0102)

keyword_0222 = ["鼻中隔", "鼻甲", "外耳道", "鼓膜", "扁桃体", "其他", "口咽黏膜", "鼻腔黏膜及分泌物"]
keyword_0222 = sort_vocab_by_length(keyword_0222)

keyword_0409 = ["病史", "既往史", "过敏史", "既往疾病史", "心脏杂音", "腹壁", "心律", "心率", "呼吸音", "肺罗音", "肺啰音"]
keyword_0409 = sort_vocab_by_length(keyword_0409)

keyword_0539 = ["阴道", "外阴", "子宫", "宫颈", "右侧附件", "左侧附件", "病史"]
keyword_0539 = sort_vocab_by_length(keyword_0539)

keyword_0709 = ["义齿", "龋齿", "缺齿", "牙周", "牙龈", "其他"]
keyword_0709 = sort_vocab_by_length(keyword_0709)

description_normal = ["未闻及", "未 闻及", "未见异常", "未减异常", "无异常", "没有异常", "正常", "未见明显异常",
                      "未减明显异常", "无明显异常", "没有明显异常", "弃查", "已告知本人"]

description_void = ["无", "未查", "弃查", "exit", "未未及"]

description_0124 = ["未见扩张", "无扩张", "没有扩张",  "未见明显扩张", "无明显扩张", "没有明显扩张"]

description_0206 = ["无压痛", "未见压痛", "没有压痛"]

description_0212 = ["无压痛", "未见压痛", "没有压痛"]

description_0421 = ["不齐", "早搏"]