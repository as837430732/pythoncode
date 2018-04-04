# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt
import operator
import time
# 读取数据
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat #点数据以及类别数据

def selectJrand(i, m):#i是第一个alpha的下标，m是所有alpha的下标
    j = i
    while (j == i):
        j = int(random.uniform(0, m))#生成一个[0, m]的随机数，int转换为整数。注意，需要import random
    return j

def clipAlpha(aj, H, L):#调整大于H小于L的alpha值
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):#初始化各参数
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))#误差缓存

def calcEk(oS, k):#计算误差
    # (alpha*lable)T * data*(data[i])T    +b
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
#选择第二个alpha
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            #选择具有最大步长的j
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    #计算误差
    Ei = calcEk(oS, i)
    # 如果误差大 则对alpha进行优化
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)#从m中选择一个随机数，第2个alpha j
        # #复制下来，便于比较  旧 新比较
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        #开始计算L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if (L == H):
            # print("L == H")
            return 0
        # eta是alphas[j]的最优修改量，如果eta为零，退出for当前循环
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            # print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta#调整alphas[j]
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): #如果alphas[j]没有调整
            # print("j not moving enough")
            return 0
        # 两个alpha同时进行改变  一个正方向  另一个是反方向
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])#调整alphas[i]
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

#中心思想  每次循环中选择两个alpha进行优化处理。一但找到一对合适的alpha 那么增大其中一个同时减小另一个
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    # 常数C 容错率  最大循环次数
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)#初始化
    iterr = 0
    entireSet = True
    alphaPairsChanged = 0#记录alpha是否已经进行优化
    while (iterr < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):#迭代次数超过指定最大值将退出循环
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):#在数据集上遍历每一个alpha
                alphaPairsChanged += innerL(i, oS)
            # print("fullSet, iter: %d i:%d, pairs changed %d" % (iterr, i, alphaPairsChanged))
            iterr += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iterr, i, alphaPairsChanged))
            iterr += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        # print("iteration number: %d" % iterr)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):

    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        #只有支持向量的alpha是非零的，因此在svm计算w中不考虑其他点
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def plotFeature(dataMat, labelMat, weights, b):
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-2, 10.0, 0.1)
    y = (-b[0, 0] * x) - 10 / linalg.norm(weights)
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def main():
    trainDataSet, trainLabel = loadDataSet('E:\\testSet.txt')
    b, alphas = smoP(trainDataSet, trainLabel, 0.6, 0.0001, 40)
    ws = calcWs(alphas, trainDataSet, trainLabel)
    print("ws = \n", ws)
    print("b = \n", b)
    plotFeature(trainDataSet, trainLabel, ws, b)

if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))