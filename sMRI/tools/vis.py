#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:48:04 2023

@author: yanshi
"""

import matplotlib.pyplot as plt
import numpy as np
def draw_line_chart(data):
    # 获取列表中的最大值和最小值
    max_value = max(data)
    min_value = min(data)

    # 计算x轴和y轴的范围
    x_range = range(len(data))
    y_range = [min_value, max_value]

    # 创建一个新的图形
    plt.figure()

    # 绘制折线图
    plt.plot(x_range, data, marker='o', linestyle='-', color='b')
    
    x = range(len(data))
    for i in range(len(x)):
        data_i="{:.5f}".format(data[i])
        plt.text(x[i], data[i], str(data_i), ha='center', va='bottom', rotation=90)
    #plt.xticks(x, rotation=90)
    # 设置x轴和y轴的标签
    plt.xlabel('pc')
    plt.ylabel('explained_variance_ratio')
    plt.tight_layout()
    #plt.xticks(np.arange(0, 31, step=5))

    # 设置x轴和y轴的范围
    plt.xlim(0, len(data))
    plt.ylim(min_value, max_value)

    # 设置x轴和y轴的刻度
    plt.xticks(x_range)
    plt.yticks([0, 1])

    # 添加标题
    plt.title('explained_variance_ratio_chart')
    
    # 显示图形
    plt.show()
if __name__ == '__main__':
    # 示例数据
    data = [1, 3, 7, 1, 2, 6, 3, 2, 4, 5]
    # 调用函数绘制折线图
    draw_line_chart(data)
