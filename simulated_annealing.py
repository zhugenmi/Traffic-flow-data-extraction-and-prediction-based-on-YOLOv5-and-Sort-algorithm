import math
import random

import numpy as np

# 25个候选设备点 5km 每200m设置一个候选点
I=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


# 监测点重要度Ni
N=[3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3]



# 通过已有监控数据计算年平均日交通量A
# avg7=1.150271575*6*60*24
# avg5=5.071759259*6*60*24
# avg8=3.725241546*6*60*24
# avg3=2.954651163*6*60*24
# for i in range(25):
#     if i < 5:
#         A.append((avg7+avg5)/2)
#     elif i <10:
#         A.append((avg5+avg8)/2)
#     else:
#         A.append((avg8+avg3)/2)
# print(A)
# 年平均日交通量Ail，Ai表示监测点i所在路段l的平均日交通量
A=[26879.173202880003, 26879.173202880003, 26879.173202880003, 26879.173202880003, 26879.173202880003, 38003.043477600004, 38003.043477600004, 38003.043477600004, 38003.043477600004, 38003.043477600004, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288, 28857.13650288]

# 道路被划分路段单元集合 道路总长度5km，划分为三段，第一段1km，第二段1km，第三段3km
L=[1,2,3]
d=[1000,1000,3000]

# 初始温度
T0=1000

# 温度下限
t_min=1e-12

# 温度下降速率
a=0.99

# 迭代次数
max_iterations=1800

# 固定建设成本3万元
cf=3

# 维修成本0.5万元
cs=0.5

# 摄像机成本1.5万元
ck=1.5

# 摄像机最大布设规模
Q=8

# 两点间最小距离
D=200

# 监控覆盖范围
r=150

# 初始温度迭代次数
K=1

# w1-经济成本权重 w2-重要性权重 w3-交通量权重
w1=0.2338
w2=0.6289
w3=0.1372

# 决策变量 xi=1表示第i个点被选中
xi=np.zeros((25), dtype=int)

# 成本目标函数
def f1(x):
    total_cost=0
    for i in range(25):
        if x[i] == 1:
            total_cost+=(cf+cs+ck)
    return w1*total_cost

# 重要性目标函数
def f2(x):
    total_imp=0
    for i in range(25):
        total_imp+=x[i]*N[i]
    return w2*total_imp

# 流量目标函数
def f3(x):
    total_flow=0
    for i in range(25):
        l=0;
        if i < 5:
            l=0
        elif i<10:
            l=1
        else:
            l=2

        total_flow+=x[i]*A[i]*r/(12*d[l])
    return w3*total_flow

# 计分函数，统一量化分值
def objective_function(x):
    return -w1*f1(x)+w2*f2(x)+w3*f3(x)

'''约束条件
1、布设监测点的数量（即xi中为1的数量）要小于最大规模Q
2、若两点被同时选中，则两点间的距离必须大于建议距离D
3、限定变量x[i]的范围，只能取0或1
4、每个路段单元至少要有1个监测点被选取
'''

# 模拟退火算法
# 约束条件检查
def check_constraints(x):
    # 1. 监测点数量
    if sum(x) > Q:
        return False
    # 2. 两点间距离
    # for i in range(25):
    #     if x[i] == 1:
    #         for j in range(i + 1, 25):
    #             if x[j] == 1 and abs(i - j) < D / 200:  # 假设每个点间隔200m
    #                 return False
    # 3. 每个路段至少一个监测点
    for i in range(25):
        if x[:5].sum() ==0:
            return False
        elif x[5:10].sum() == 0:
            return False
        elif x[10:].sum() ==0:
            return False
    return True


# 模拟退火算法
def simulated_annealing():
    # 初始化
    current_solution = np.random.randint(2, size=25)
    while not check_constraints(current_solution):
        current_solution = np.random.randint(2, size=25)

    current_score = objective_function(current_solution)
    best_solution = current_solution.copy()
    best_score = current_score

    T = T0
    iteration = 0

    while T > t_min and iteration < max_iterations:
        # 生成新解
        new_solution = current_solution.copy()
        idx = random.randint(0, 24)
        new_solution[idx] = 1 - new_solution[idx]  # 翻转某个点的选择状态

        if check_constraints(new_solution):
            new_score = objective_function(new_solution)
            delta_E = new_score - current_score

            # Metropolis准则
            if delta_E > 0 or random.random() < math.exp(delta_E / T):
                current_solution = new_solution
                current_score = new_score

                if current_score > best_score:
                    best_solution = current_solution.copy()
                    best_score = current_score
                    print('solution:{}  score:{}'.format(best_solution,best_score))

        T *= a  # 降温
        iteration += 1

    return best_solution


# 执行模拟退火算法
best_solution = simulated_annealing()
print("Best solution found:", best_solution)

for i in range(25):
    if best_solution[i] == 1:
        print(i+1)

# print('{} {} {} {}'.format(best_solution[0],best_solution[4],best_solution[9],best_solution[24]))