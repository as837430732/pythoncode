# -*- coding: utf-8 -*-
from numpy import *;#导入numpy的库函数
import numpy as np; #这个方式使用numpy的函数时，需要以np.开头。
class test2(object):
    #A三次方
    def mul(self,A):
        return pow(A,3)
    def ni(self,A):
        A2=A.I
        return A2
    def hls(self,A):
         return np.linalg.det(A)
    def tzz(self,A):
        return np.linalg.eig(A)

#A= mat( random.rand(3,3) )
#t= test2()   
#print('A的三次方  ')
#print(t.mul(A))
#
#print('A的逆矩阵 ')
#print(t.ni(A))
#
#print('A的行列式值')
#print(t.hls(A))
#
#print('A的特征值')
#print(t.tzz(A))