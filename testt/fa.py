from click._compat import raw_input
from matplotlib import pyplot

#基于间隔的算法
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
    # train_data1 = [[1, 3, 1], [2, 5, 1], [3, 8, 1], [2, 6, 1]]  # 正样本
    # train_data2 = [[3, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]  # 负样本
    train_data = [[1, 1, -1], [1, 2, -1], [2, 1, -1], [3, 3, -1], [1, 4, -1], [5, 7, 1], [4, 5, 1], [5, 6, 1],
                    [6, 5, 1]]
    #train_data=train_data1+train_data2
    weight = [0, 0]  # 权重
    bias = 0  # 偏置量
    learning_rate = 0.1 # 学习速率

    train_num = int(raw_input("train num: "))  # 迭代次数

    # 显示图片
    mk = []
    cs = []
    for t in range(len(train_data)):
        if train_data[t][2] > 0:
            mk.append('o')
            cs.append('red')
        else:
            mk.append('x')
            cs.append('blue')
    x1, x2, y = zip(*train_data)#将数据分开分别放入x1 x2 y中
    print(x1, x2)
    for _s, _c, _x1, _x2 in zip(mk, cs, x1, x2):
        pyplot.scatter(_x1, _x2, marker=_s, c=_c, s=100)
    # pyplot.scatter(x1, x2, color=cs, marker=mk)
    pyplot.text(6.8, 6.8, r'+1', fontdict=fontpo1)
    pyplot.text(0.2, 0.2, r'-1', fontdict=fontpl1)
    pyplot.axis([0, 9, 0, 9])
    pyplot.plot([4, 3], [0, 9], 'k-', linewidth=2.0, color='green')
    pyplot.show()

    for z in range(train_num*10000):
        for i in range(len(train_data)):
            predict = sign(weight[0] * x1[i] + weight[1] * x2[i] + bias)  # 输出

            print("train data: x: (%d, %d) y: %d  ==> predict: %d" % (x1[i], x2[i], y[i], predict))
            if (weight[0]*x1[i]+weight[1]*x2[i]) * y[i] < 1:  # 判断误分类点
                t1= (1-y[i]*x1[i]*weight[0])/(x1[i]**2)
                t2 = (1 - y[i] * x2[i] * weight[1]) / (x2[i] ** 2)
                print(t1,t2,"dfdfdfd")
                weight[0] = weight[0] +   y[i] * x1[i]*t1  # 更新权重
                weight[1] = weight[1] +   y[i] * x2[i]*t2
                bias = bias +   y[i]*(t2+t1)  # 更新偏置量
                print("update weight and bias: "),
                print(weight[0], weight[1], bias)
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
    # print(s1)
    if s1[1] > 9 or s1[1] < 0:
        s1 = ((9 * weight[1] + bias) / -weight[0], 9)
        # print(s1)
    s2 = (-bias / weight[0], 0)
    if s2[0] > 9 or s2[0] <= 0:
        s2 = (9, (9 * weight[0] + bias) / -weight[1])
        print(s2)
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