import matplotlib.pyplot as plt
from random import *
import numpy as np
from functools import reduce
import math
x=[]
y=[]
sigma=0.2
num=100
c=200
rand = Random()
rand.seed(1)  # seed() 方法改变随机数生成器的种子
#  用正态分布作为随机参数
# μ是均值,标准差σ   extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
x.extend([rand.normalvariate(1, sigma) for _ in range(num)])  # 循环产生100个点
x.extend([rand.normalvariate(1, sigma) for _ in range(num)])
x.extend([rand.normalvariate(2, sigma) for _ in range(num)])
x.extend([rand.normalvariate(2, sigma) for _ in range(num)])
x.extend([rand.normalvariate(3, sigma) for _ in range(num)])
x.extend([rand.normalvariate(3, sigma) for _ in range(num)])

y.extend([rand.normalvariate(1, sigma) for _ in range(num)])
y.extend([rand.normalvariate(2, sigma) for _ in range(num)])
y.extend([rand.normalvariate(2, sigma) for _ in range(num)])
y.extend([rand.normalvariate(3, sigma) for _ in range(num)])
y.extend([rand.normalvariate(1, sigma) for _ in range(num)])
y.extend([rand.normalvariate(3, sigma) for _ in range(num)])
#miux miuy pi sigmax sigmay cov初始化
m_x=[1.5, 2.0, 1.5,2.0,2.5,2.5]#中心点
m_y=[2.0, 1.0, 1.0,2.5,1.0,3.0]
pi = [3.14,3.14,3.14,3.14,3.14,3.14]
sigma_x=[0.5,0.5,0.5,0.5,0.5,0.5]#方差
sigma_y=[0.6,0.6,0.6,0.6,0.6,0.6]
cov = [-0.5,-0.5,-0.5,-0.5,-0.5,-0.5]
def add(x,y):
    return x+y


def gauss(x,y,m_x,m_y,pi,sigma_x,sigma_y,cov):
    D=np.matrix([[sigma_x, cov], [cov, sigma_y]])#协方差矩阵
    n_D=np.linalg.pinv(D)  # 协方差的逆
    h_D= np.linalg.det(n_D)  # 协方差的行列式
    if h_D < 0:
        h_D = 0
    u=np.matrix([[x - m_x], [y - m_y]])
    z_u=np.transpose(u) #u的转置

    t= 1 / (2 * np.math.pi) * np.sqrt(np.abs(h_D))
    tt= math.exp(-0.5 * (z_u * n_D * u))
    # print(t*tt)
    return t*tt

#设置颜色
color_set = ['red','green','yellow','blue','pink','purple']
plt.scatter(x, y, color='black')
plt.scatter(m_x, m_y, color=color_set, s=250, alpha=0.7) #s半径大小
plt.show()


p=[[0, 0, 0,0,0,0] for a in range(len(x))]
#对p进行初始化 p代表的每个类的概率
for n in range(len(x)):
    # print(n)
    for i in range(len(m_x)):
        # print(i)
        p1 = pi[i]*gauss(x[n],y[n],m_x[i],m_y[i],pi[i],sigma_x[i],sigma_y[i],cov[i])

        p2=reduce(add,[
                pi[t]*gauss(x[n],y[n],m_x[t],m_y[t],pi[i],sigma_x[t],sigma_y[t],cov[t])
                for t in range(len(m_x))])
        p[n][i]=p1/p2
# for pr in p:
#     print(type(pr))
# print(type(p))
# print(p[11][0])
for nc in range(c):
        # 更新高斯函数参数
    for k in range(len(m_x)):
        s = reduce(add, [p[i][k] for i in range(len(x))])
        pi[k]=reduce(add,[p[i][k] for i in range(len(x))])/len(x)
        m_x[k] = reduce(add,[x[i]*p[i][k] for i in range(len(x))])/s
        m_y[k] = reduce(add, [y[i] * p[i][k] for i in range(len(y))]) / s
        sigma_x[k]= reduce(add,[p[i][k]*((x[i]-m_x[k])**2) for i in range(len(x))])/s
        sigma_y[k] = reduce(add, [p[i][k] * ((y[i] - m_y[k]) ** 2) for i in range(len(y))]) / s
        cov[k] = reduce(add,[p[i][k]*(x[i]-m_x[k])*(y[i]-m_y[k]) for i in range(len(x))])/s

        #更新p中的概率
    for n in range(len(x)):
            # print(n)
            for i in range(len(m_x)):
                # print(i)
                p1 = pi[i] * gauss(x[n], y[n], m_x[i], m_y[i], pi[i], sigma_x[i], sigma_y[i], cov[i])

                p2 = reduce(add, [
                    pi[t] * gauss(x[n], y[n], m_x[t], m_y[t], pi[i], sigma_x[t], sigma_y[t], cov[t])
                    for t in range(len(m_x))])
                p[n][i] = p1 / p2
    if nc%5==0:
        z_color_list = []
        for i1 in range(len(x)):
            z_color_list.append(color_set[p[i1].index(max(p[i1]))])
        plt.scatter(m_x, m_y, color=['red','green','yellow','blue','pink','purple'], s=100)
        plt.scatter(x, y, color=z_color_list)
        plt.show()


