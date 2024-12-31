# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 07:26:07 2024

@author: 17664
"""

import os
import sys
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, cpu_count


#---------------------------- Direction Settings ------------------------------

current_file_path = os.path.abspath(__file__)       # 获取当前文件的绝对路径
current_dir = os.path.dirname(current_file_path)    # 获取当前文件的目录
parent_dir = os.path.dirname(current_dir)           # 获取当前文件目录的父目录
sys.path.append(parent_dir)                         # 将 'parent_dir' 目录添加到 sys.path
#----------------------------------- Settings ---------------------------------
from multicouptop import *
from multicouptop.Settings import fig_width,fig_height



#================================== Parameters ================================    #b1 = (np.sqrt(5)-1)/2
BC = 1                                            # 0为全开边界，1为全周期性边界
S = 4                                             # 链数
L = 55                                            # 单条链的长度
t1 = 1.0                                          # 水平跃迁
tx = 0.2                                          # 交叉跃迁
t2 = 0                                            # 垂直跃迁
V_0 = 1.1                                         # 无序势强度
lam_array = [0]                                   # 水平跃迁的准周期调制
H_size = S*L                                      # 哈密顿量矩阵的尺寸
phi1 =0                                           # 水平跃迁的相位         
phi3_array = np.arange(0,2*np.pi,0.004)           #无序势的相位                        
E0_array = np.array([0,5])                        # 计算绕数时，能谱包围的中心点。



#================================ Calculations ================================


def compute_E_translation(params):    
    i, j, lam_array, phi3_array, L, tx, t1, t2, V_0, E0_array = params
    translation = E0_array[0]
    lam = lam_array[i]
    phi3 = phi3_array[j]
    # 创建 Hamiltonian 类的实例
    hamiltonian_instance = Hamiltonian(t1,lam, phi1, t2, tx, V_0, L, S, BC, phi3, translation)              #利用导入的Hamiltonian类定义和本文件的变量，定义本文件的Hamiltonian类。
    # 通过实例调用 get_hamiltonian 方法
    H33 = hamiltonian_instance.get_hamiltonian(L, S, t1, t2, tx, V_0, lam, phi3/H_size, BC, translation)    #利用本文件的Hamiltonian类下的Hamiltonian方法，得到特定哈密顿量H。    
    det3 = np.linalg.det(H33)
    return det3



if __name__ == "__main__":
    params = [(i, j, lam_array, phi3_array, L, tx, t1, t2, V_0, E0_array) \
              for i in range(len(lam_array)) for j in range(len(phi3_array))]
    with Pool(processes = cpu_count()) as pool:   
        # 使用tqdm的imap方法来添加进度条
        results = list(tqdm(pool.imap(compute_E_translation, params), total=len(params)))
    # 将结果转换回二维数组形式   
    det3_results = results      
    det3_array = np.array(det3_results)
    
    
            

    #================================== Plot ======================================
    fig_index = 1
        # Plot
    fig, ax = plt.subplots(figsize=(fig_width,fig_height))
    ax.tick_params(labelsize=32)
    plt.figure(fig_index)
    fig_index += 1
    sc = plt.plot(phi3_array,np.angle(det3_array),'-o')
    plt.title(f'V₀={V_0},$t_x$={tx},  L={L}, ChainNumber={S},BC={BC}, $V(i)=V_0*e^{{1.0j*(2*\\pi*\\alpha*(i-(i//L)*L+1)+\\phi_3)}}$', fontsize=20)    #phi3={phi3_array[j]:.2f},
    plt.xlabel('flux',fontsize=42)
    plt.ylabel('argument',fontsize=42)
    plt.tight_layout() 
    plt.savefig(fr'{current_dir}\ω-V_save\V={V_0_array[j]:.2f}.png')
    
    
    
    
    
    
    