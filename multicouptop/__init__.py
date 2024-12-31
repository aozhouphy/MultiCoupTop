# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 07:31:57 2024

@author: 17664
"""


# __init__.py    # 用于将一个目录标记为包的文件。同时做初始化。
#__all__ = ['Hamiltonian']

# multicouptop/__init__.py
#定义包的接口，例如定义一些函数、类或变量# settings/__init__.py from .Settings import some_function, SomeClass, some_variable；当从包名导入时，可以直接导入这些定义：from settings import some_function, SomeClass, some_variable
from .Hamiltonian import Hamiltonian    #   从文件导入类。  
from . import Settings                  #   从目录导入整个文件。


