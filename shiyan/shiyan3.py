
# 符号函数
import random
#感知机算法



from click._compat import raw_input
from matplotlib import pyplot

fontpo1 = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 16,
        }

fontpl1 = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 16,
        }


def sign(v):
    if v > 0:
        return 1
    else:
        return -1

def training():
     train_data1 = [[1, 3, 1], [2, 5, 1], [3, 8, 1], [2, 6, 1]]  # 正样本
     train_data2 = [[3, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]  # 负样本
     #train_data1=[]
     #train_data2=[[1,1,-1],[1,2,-1],[2,1,-1],[3,3,-1],[1,4,-1],[5,7,1],[4,5,1],[5,6,1],[6,5,1]]

     train_datas = train_data1 + train_data2  # 样本集

     weight = [0, 0]  # 权重
     bias = 0  # 偏置量
     learning_rate = 1.0  # 学习速率

     #print(str(train_datas[0][2])+"dsdsdsss")
     train_num = int(raw_input("train num: "))  # 迭代次数

     #显示图片
     mk = []
     cs = []
     for t in range(len(train_datas)):
             if train_datas[t][2] > 0:
                mk.append('o')
                cs.append('red')
             else:
                mk.append('x')
                cs.append('blue')
     x1, x2,y = zip(*train_datas)
     print(x1,x2)
     for _s, _c, _x1, _x2 in zip(mk, cs, x1, x2):
         pyplot.scatter(_x1, _x2, marker=_s, c=_c, s=100)
     # pyplot.scatter(x1, x2, color=cs, marker=mk)
     #pyplot.text(0.2, 6.8, 'haoran', fontdict=fontpo1)
     pyplot.text(6.8, 6.8, r'+1', fontdict=fontpo1)
     pyplot.text(0.2, 0.2, r'-1', fontdict=fontpl1)
     pyplot.axis([0, 9, 0, 9])
     pyplot.plot([4, 3], [0, 9], 'k-', linewidth=2.0, color='green')
     pyplot.show()

     for ii in range(len(train_datas)*10):
         for la in range(len(train_datas)):
             # for laa in range(len(train_datas)):
                 for i in range(len(x1)):
                     # train = random.choice(train_datas)
                     # x1, x2, y = train

                     predict = sign(weight[0] * x1[i] + weight[1] * x2[i] + bias)  # 输出
                     print("train data: x: (%d, %d) y: %d  ==> predict: %d" % (x1[i], x2[i], y[i], predict))
                     if y[i] * predict <= 0:  # 判断误分类点
                         weight[0] = weight[0] + learning_rate * y[i] * x1[i]  # 更新权重
                         weight[1] = weight[1] + learning_rate * y[i] * x2[i]
                         bias = bias + learning_rate * y[i]  # 更新偏置量
                         print("update weight and bias: "),
                         print(weight[0], weight[1], bias)
                         # history.append([w, b])  # 存储w，b






     print("stop training: "),
     print(weight[0], weight[1], bias)
     for i in range(len(x1)):
         s = sign(weight[0] * x1[i] + weight[1] * x2[i] + bias)  # 输出
         if s > 0:
             cs[i] = 'red'
             mk[i] = 'x'
         else:
             cs[i] = 'blue'
             mk[i] = 'o'

     for _s, _c, _x1, _x2 in zip(mk, cs, x1, x2):
         pyplot.scatter(_x1, _x2, marker=_s, c=_c, s=100)

     pyplot.axis([0, 9, 0, 9])
     s1 = (0, -bias / weight[1])
     #print(s1)
     if s1[1] > 9 or s1[1] < 0:
         s1 = ((9 * weight[1] + bias) / -weight[0], 9)
         #print(s1)
     s2 = (-bias / weight[0], 0)
     if s2[0] > 9 or s2[0] <= 0:
         s2 = (9, (9 * weight[0] + bias) / -weight[1])
         print(s2)
     pyplot.text(0.2, 8.0, 'haoran', fontdict=fontpo1)
     pyplot.plot([s1[0], s2[0]], [s1[1], s2[1]], 'k-', linewidth=2.0, color='green')
     print(s1[0], s2[0], s1[1], s2[1], "234423")
     pyplot.show()


     return weight, bias

def test():
    weight, bias = training()
    # while True:
    #     test_data = []
    #     data = raw_input('enter test data (x1, x2): ')
    #     if data == 'q': break
    #     test_data += [int(n) for n in data.split(',')]
    #     predict = sign(weight[0] * test_data[0] + weight[1] * test_data[1] + bias)
    #     print("predict ==> %d" % predict)

test()
