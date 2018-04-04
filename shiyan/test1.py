# -*- coding: utf-8 -*-
import math

#abab
class test1(object):
    def __init__(self,string):
        self.string = string
    def test1_count(self):
        n=len(self.string)
        while n>0:
            n=n-1
            if self.string.count(self.string[n])!= self.string.count(self.string[n-1]):
                return 0
        return 1
    def print_test1(self,flag):
        if flag==1:
            print('%s  满足' % (self.string))
        if flag==0:
             print('%s  不满足' % (self.string))

#b = test1('DDFF')
#
#b.print_test1(b.test1_count())

