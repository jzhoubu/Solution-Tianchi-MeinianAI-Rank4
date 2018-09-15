# -*- coding:utf-8 -*-

class ColumnDataTypeError(Exception):
    def __init__(self, err='列数据类型错误'):
        Exception.__init__(self, err)