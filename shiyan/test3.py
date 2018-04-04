# -*- coding: utf-8 -*-
class test3(object):
    def __init__(self):
        pass

    def ji(self, n):
        factorial = 1
        # 查看数字是负数，0 或 正数
        if n < 0:
            print("抱歉，负数没有阶乘")
            return 0
        elif n == 0:
            return 1
        else:
            for i in range(1, n + 1):
                factorial = factorial * i

            return factorial

    def co(self, X, n):
        t = test3()
        sum = 0.0
        for i2 in range(0, n):
             sum += (X ** i2) / t.ji(i2)

        print('e的' + str(X) + '次方   ' + str(sum))
# 1+x/1!+x^2/2!+...+x^n/n!






    # print("E^" + str(X) + " = " + str(E))
