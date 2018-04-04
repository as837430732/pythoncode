# -*- coding: utf-8 -*-
import shiyan.test1
# import imp
# util = imp.load_source('util', 'E:\pythoncode\shiyan\test1.py')
import re
from shiyan.test1 import test1
from shiyan.test2 import test2
from shiyan.test3 import test3
from shiyan.test4 import test4
from shiyan.test5 import test5

from numpy import *;#导入numpy的库函数
import numpy as np; #这个方式使用numpy的函数时，需要以np.开头。


print('选择功能：')
number = input()
if(number=='1'):
     file_object = open('E:\AI\data\one_test')
     try:
         all_the_text = file_object.read()
     finally:
         file_object.close()
     b=[]
     b=all_the_text.split('\n')
     print(b[0])
     for it in b:
        t = test1(it)
        t.print_test1(t.test1_count())
         
elif(number=='2'):
    file_object = open('E:\AI\data\\two_test')
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()
    f1=0
    l=[]
    #list=[1]*50
    list=''
    l=all_the_text.split('\n')
    print(l)
    print(l[1])
    for i in range(len(l)):

        if l[i]=='':
           list=list+','
        else:
           list=list+str(l[i])+';'
          # f1=f1+1

    print(list)
    c=list.split(',')
    for i2 in range(len(c)-1):
       a=np.asmatrix(c[i2][:-1])
       print('a\n', a)
       print('a^3\n', mat(a)**3)
       print('逆矩阵\n', np.linalg.inv(a))
       print('特征值\n', np.linalg.eigvals(a))
       print('行列式\n', np.linalg.det(a))
   # # st = re.compile('_')
   #  a = list.replace(" ", "  ")
   #  b=a.replace("_", "[ ")
   #  c=b.replace("-", "]\n")
   #
   #
   #  # print(c)
   #
   #  re=[]
   #  re=c.split(',')
   #
   #  print(re)
   #  print(random.rand(3, 3))
   #
   #  print(mat(array('['+re[1][:-1]+']')))
   #
   #  A = mat(random.rand(3, 3))
   #  print(A)
    # for i2 in range(1, len(re)-1):
    #     A=mat('['+re[i2][:-1]+']')
    #     t=test2()
    #     print('A的三次方  ')
    #     print(t.mul(A))

    # l2=list.split(',')
    # for i2 in range(2len(l2)):
    #     for i3 in range(len(l2[i2])):
    #         if l2[i2][i3]=='-':

   # print(l2)
   # A= mat( random.rand(3,3) )
   # t= test2()
   # print('A的三次方  ')
   # print(t.mul(A))
   #
   # print('A的逆矩阵 ')
   # print(t.ni(A))
   #
   # print('A的行列式值')
   # print(t.hls(A))
   #
   # print('A的特征值')
   # print(t.tzz(A))
elif(number=='3'):
  t = test3()
  t.co(1, 100)

elif(number=='4'):
    t = test4()
    t.tes()
elif(number=='5'):
    #编码 格式必学为 utf-8中文的缘故
    file_object = open('E:\AI\data\\five_test','r',encoding= 'utf-8')
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()
    t = test5()
    t.fc(all_the_text)
