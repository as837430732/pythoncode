from collections import OrderedDict

class test5(object):
    def fc(self, s):
        s=s.replace('\n', ' ')
        list = s.split(' ')
        # 数据字典
        letterCounts = OrderedDict([])

        for letter in list:
            #print(letter)
            letterCounts[letter] = letterCounts.get(letter, 0) + 1
            # print(letter+str(letterCounts.get(letter, 0) + 1))

       # print(letterCounts)

        bar = OrderedDict(sorted(letterCounts.items(), key=lambda x: x[1], reverse=True))

        # print(sorted(letterCounts.items()))
        print(bar)


