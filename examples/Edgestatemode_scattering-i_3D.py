# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:09:39 2024

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
from mpl_toolkits.mplot3d import Axes3D


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
L = 233                                           # 单条链的长度
t1 = 1.0                                          # 水平跃迁
tx = 0.2                                          # 交叉跃迁
t2 = 0                                            # 垂直跃迁
V_0_array = np.arange(1.4, 1.6*t1+0.0001, 0.08)   # 无序势强度
lam_array = [0]                                   # 水平跃迁的准周期调制
H_size = S*L                                      # 哈密顿量矩阵的尺寸
phi1 = phi3 = 0                                   # 水平跃迁、无序势的相位
translation = 0                                   # 计算绕数时，能谱包围的中心点。
LS = 0.90                                         # 局域态判定条件中的系数


#================================ Calculations ================================


def compute_E(params):    #compute_MIPR(params):
    i, j, lam_array, V_0_array, L, tx, t1, t2, translation = params    #i
    lam = lam_array[i]
    V_0 = V_0_array[j]    
    # 创建 Hamiltonian 类的实例
    hamiltonian_instance = Hamiltonian(t1,lam, phi1, t2, tx, V_0, L, S, BC, phi3, translation)     #利用导入的Hamiltonian类定义和本文件的变量，定义本文件的Hamiltonian类。
    # 通过实例调用 get_hamiltonian 方法
    H = hamiltonian_instance.get_hamiltonian(L, S, t1, t2, tx, V_0, lam, phi3, BC, translation)    #利用本文件的Hamiltonian类下的Hamiltonian方法，得到特定哈密顿量H。    
    ev, es = np.linalg.eig(H)
    sorted_indices = np.argsort(ev)
    E, ES = ev[sorted_indices], es[:,sorted_indices]
    P = np.conj(ES)*ES
    IPR = np.sum(P**2,axis=0)/(np.sum(P,axis=0)**2)
    MIPR = np.sum(IPR,axis=0)/H_size
    return E,ES,IPR,MIPR



if __name__ == "__main__":    
    params = [(i, j, lam_array, V_0_array, L, tx, t1, t2, translation) for i in range(len(lam_array)) for j in range(len(V_0_array))]

    with Pool(processes = cpu_count()) as pool:
        # 使用tqdm的imap方法来添加进度条
        results = list(tqdm(pool.imap(compute_E, params), total=len(params)))
    # 将结果转换回二维数组形式
    E_results, ES_results, IPR_results, MIPR_results = zip(*results)    #zip函数解压是一一交错搭配的。[[E1,IPR1],[E2,IPR2],[E3,IPR3],...] --> [E1,E2,E3,...],[IPR1,IPR2,IPR3,...]
    E_array, ES_array, IPR_array, MIPR_array = np.array(E_results), np.array(ES_results), np.array(IPR_results), np.array(MIPR_results).reshape(len(lam_array), len(V_0_array))      
    E_array_real, E_array_imag = np.real(E_array), np.imag(E_array)
    
    
    
    # 找到每个本征矢中，在格点上的分量中最大的那个，还有第二、第三大的那两个，并返回索引。
    #如果最大的那个比第二、第三个之和还要大1倍，则该本征矢局域。
    # 如果索引index是 sL-1或 sL中的任意一个，且0≤S≤L，且0≤index≤SL-1，以及index在[0，L-2]和[(S-1)*L-1,LS-2]，则该本征矢为局域边缘态。
    
    edge_site_list = []
    for s in range(1,S,1):
        edge_site_list.append(s*L-1)
        edge_site_list.append(s*L)
    for i in range(1,L,1):
        edge_site_list.append((S-1)*L+i)
        edge_site_list.append((0*L+i-1))
    
    
    P_array = ES_array*np.conj(ES_array)
    
    fig_index = 1
    for j in range(len(V_0_array)):    #索引为j的参数V_0
        V_0 = V_0_array[j]             
        P_V_0 = P_array[j,:,:]       #索引为j的参数V_0对应的所有本征矢
        for q in range(0,H_size,1):    #取出其中第q个本征矢
            P_q = P_V_0[:,q]          #取出其中第q个本征矢
            index_array = np.argsort(P_q)
            max_0 = P_q[index_array[-1]]
            max_1 = P_q[index_array[-2]]
            max_2 = P_q[index_array[-3]]
            if max_0 > LS*(max_1+max_2):    # 满足这个条件的认为是局域态（Localized State）
               if index_array[-1] in edge_site_list:
                   # Plot
                    fig, ax = plt.subplots(figsize=(fig_width,fig_height))
                    ax.tick_params(labelsize=32)
                    plt.figure(fig_index)
                    fig_index += 1
                    site_index = np.arange(0,H_size,1)
                    # 绘制柱状图
                    ax = fig.add_subplot(111, projection='3d')
                    dx = dy =1                                       #x，y两轴的值的最小间距，由于格点最小单位为1，故取1.
                    for i in range(H_size):
                        ax.bar3d((site_index-(site_index//L)*L)[i], (site_index//L)[i], 0, dx, dy, P_q[i], color='b')
                    plt.title(f'V={V_0_array[j]:.2f},$t_x$={tx}, L={L}, ChainNumber={S},BC={BC}, $V(i)=V_0*e^{{1.0j*(2*\\pi*\\alpha*(i-(i//L)*L+1)+\\phi_3)}}$, localized at site {index_array[-1]}', fontsize=20)
                    plt.xlabel('$Site_x$',fontsize=42)
                    plt.ylabel('$site_y$',fontsize=42)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    