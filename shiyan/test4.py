#思路 一行行读文件 没两行存入到两个list数组中 写一个方法此方法主要用来判断小括号 大括号，当为小括号时则
class test4(object):
    def __init__(self):
        pass
    def floatrange(start,stop,steps):
        return [start + float(i) * (stop - start) / (float(steps) - 1) for i in range(steps)]

    def two(self,l1, l2):
        s = l1.split(',')
        s1 = l2.split(',')

        a = set()
        for i in range(len(s)):
            a.add(s[i])
        # print(a)

        b = set()
        for i2 in range(len(s1)):
            b.add(s1[i2])
        # print(b)
        # a=[]
        # a=list1[0].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', ',')
        # print(a)
        # b=list1[1].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', ',')
        print("交"+str(a & b))
        print("并"+str(a | b))
        t=('').join(a | b).replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').replace('  ', ' ')
        print(t)
        t1= t.split(' ')
        la=set()
        for i in range(1,len(t1)-1):
            la.add(t1[i])
        print(la)
        z = [float(i) for i in la]
        print(max(z))
        print(min(z))
        for i2 in range(int(min(z)),int(max(z))):
            print(str(i2))

    def tes(self):
        file_object = open('E:\AI\data\\four_test')
        try:
            all_the_text = file_object.read()
        finally:
            file_object.close()


        list1=all_the_text.split('\n')
        print(list1)
        print(('').join(list1))
        for i in range(len(list1)-1):
            if i%3==0:
                self.two(list1[i], list1[i + 1])

    # def sel(l1,l2):
    #     m = 0
    #     n = len(('').join(l1))
    #     E = []
    #     b = 0
    #     while (m < n):
    #         i = n - m
    #         while (i >= 0):
    #             E.append(('').join(l1)[m:m + i])
    #             i -= 1
    #         m += 1
    #
    #     for x in E:
    #         a = 0
    #         if x in ('').join(l1):
    #             a = len(x)
    #             c = E.index(x)
    #         if a > b:  # 保存符合要求的最长字符串长度和地址
    #             b = a
    #             d = c
    #
    #     if b > 0:
    #         print(E[d])

        # s=('').join(list1[0])+'~'+('').join(list1[1])
        # print(s)
        #
        # a=list1[0].replace('(','').replace(')','').replace('[','').replace(']','').replace(' ',',')
        # # print(list1[0])
        # b = list1[1].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', ',')
        # # print(list1[1])
        # print(list(set(a).intersection(set(b))))
        # print(list(set(a).union(set(b))))
