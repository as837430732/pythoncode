import matplotlib.pyplot as plt
from random import *

# 改程序的思想：首先通过高斯分布（正态分布）随机生成300个点 并且定义好K个类别
#  然后遍历每个点并计算各点到K类点的距离，将每个点分类
#  将属于同一类别的点进行坐标平均值的计算，并将计算结果当作新的K个类别点  以此类推
num = 100
train_steps = 100
sigma = 0.2#标准差

random_x = []
random_y = []
y = []
rand = Random()
rand.seed(1) #seed() 方法改变随机数生成器的种子

#用正态分布作为随机参数
# μ是均值,标准差σ   extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
random_x.extend([rand.normalvariate(1, sigma) for _ in range(num)]) #循环产生100个点
random_x.extend([rand.normalvariate(1, sigma) for _ in range(num)])
random_x.extend([rand.normalvariate(2, sigma) for _ in range(num)])
random_x.extend([rand.normalvariate(2, sigma) for _ in range(num)])
random_x.extend([rand.normalvariate(3, sigma) for _ in range(num)])
random_x.extend([rand.normalvariate(3, sigma) for _ in range(num)])

random_y.extend([rand.normalvariate(1, sigma) for _ in range(num)])
random_y.extend([rand.normalvariate(2, sigma) for _ in range(num)])
random_y.extend([rand.normalvariate(1, sigma) for _ in range(num)])
random_y.extend([rand.normalvariate(3, sigma) for _ in range(num)])
random_y.extend([rand.normalvariate(1, sigma) for _ in range(num)])
random_y.extend([rand.normalvariate(4, sigma) for _ in range(num)])

# y.extend([0 for _ in range(num)])
# y.extend([1 for _ in range(num)])
# y.extend([2 for _ in range(num)])

#显示随机的600个点
plt.scatter(random_x, random_y, color='black')
plt.show()

#设置初始化的n个点 这个是分成n类的意思 这里是分成三类
pred_x = [0.2, 0.3, 0.4,0.5,0.6,0.7]
pred_y = [0.2, 0.3, 0.4,0.5,0.6,0.7]

#设置颜色
color_set = ['red','green','yellow','blue','pink','purple']
plt.scatter(random_x, random_y, color='black')
plt.scatter(pred_x, pred_y, color=color_set, s=250, alpha=0.7) #s半径大小
plt.show()

label = []#其中记录了每个点属于哪一类中心点  例如（1,1）属于中心点1

#开始循环更新
for step in range(train_steps):
    print("Step=",step)
    for i in range(len(random_x)):#遍历所有点
        min_j, min_dist = -1, -1
        for j in range(len(pred_x)):#在三个点中选择距离点
            dist = pow(random_x[i]-pred_x[j], 2) + pow(random_y[i]-pred_y[j], 2)#距离公式
            if min_j == -1 or dist < min_dist:
               min_j, min_dist = j, dist
        if step == 0:
            label.append(min_j)
            #这里是通过step等于0时将label集合长度增加为300 方便后续的赋值
        else:
            label[i] = min_j


    for i in range(len(pred_x)):
        count, x_sum, y_sum = 1, pred_x[i], pred_y[i]
        for j in range(len(random_x)):
            if label[j] == i:
                count += 1
                x_sum += random_x[j]
                y_sum += random_y[j]
        pred_x[i], pred_y[i] = x_sum/count, y_sum/count

    #show更新的图
    pred_colors = [color_set[k] for k in label] #给点分类的颜色
    # plt.scatter(random_x, random_y, color=pred_colors)
    # plt.scatter(pred_x, pred_y, color=color_set, s=250, alpha=0.7)
    # plt.show()

# 显示最后的分类
data_colors = [color_set[l] for l in label]
plt.scatter(random_x, random_y,color= data_colors, alpha=0.8)
plt.show()#自己查看效果
