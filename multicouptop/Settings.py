# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 02:17:50 2024

@author: 17664
"""
from matplotlib import rcParams

#----------------------------- numpy settings ---------------------------------

# 将Numpy修改为单核运行。
import os
import sys

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
#"""



#---------------------------------- Plot Settings ----------------------------------

config = {
    "font.family":'serif',        
    "mathtext.fontset":'stix',    
    #"font.serif": ['SimSun'],     
    "axes.unicode_minus": False, 
    "xtick.direction":'in',       
    "ytick.direction":'in',      
}
rcParams.update(config)



from screeninfo import get_monitors
# 获取屏幕尺寸
monitor = get_monitors()[0]  # 假设只有一个屏幕
screen_width = monitor.width
screen_height = monitor.height*0.88    #给title留些位置，所以乘以了0.88

# 设置图表尺寸（减去一些边距以避免任务栏或其他界面元素）
fig_width = screen_width / 100  # 例如，假设屏幕宽度的100分之一作为边距
fig_height = screen_height / 100  # 同样，假设屏幕高度的100分之一作为边距





