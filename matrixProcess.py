import math
import numpy
from scipy import stats


def createMatrix(dims):  # 创建矩阵，输入int则输出一维，输入list输出多维
    if isinstance(dims, int):
        outputMatrix = numpy.zeros(shape=dims, dtype=float)
    else:
        shapeTuple = ()
        for dim in list(dims):
            shapeTuple = shapeTuple + (int(dim),)
        outputMatrix = numpy.zeros(shape=shapeTuple, dtype=float)
    return outputMatrix


def maxNorm(matrix: numpy.ndarray, dim: int):
    maxValue = numpy.max(matrix, axis=dim, keepdims=True)
    return matrix / maxValue


def sumNorm(matrix: numpy.ndarray, dim: int):
    sumValue = numpy.sum(matrix, axis=dim, keepdims=True)
    return matrix / sumValue


def zscore(matrix:numpy.ndarray):
    minValue = numpy.min(matrix,axis=0,keepdims=True)
    maxValue = numpy.max(matrix, axis=0, keepdims=True)
    outputMatrix = (matrix-minValue)/(maxValue-minValue)
    return outputMatrix


def EuclideanDistanceMatrix(vector: numpy.ndarray):  # 向量元素相互之间欧氏距离
    distanceMatrix = numpy.zeros(shape=(len(vector), len(vector)), dtype=float)
    for i in range(len(distanceMatrix)):
        for j in range(len(distanceMatrix[0])):
            distanceMatrix[i][j] = 1 / (math.sqrt((vector[i] - vector[j]) ** 2) + 1)
    outputMatrix = numpy.sum(distanceMatrix, axis=1)
    return outputMatrix


def dispersionMatrix(matrixIn: numpy.ndarray):  # 离差率
    outputMatrix = numpy.zeros((matrixIn.shape[0], matrixIn.shape[1]), dtype=float)
    for i in range(matrixIn.shape[0]):
        for j in range(matrixIn.shape[1]):
            outputMatrix[i][j] = 1 - (matrixIn[i][j][2] - matrixIn[i][j][0]) / (2 * matrixIn[i][j][1])
    return outputMatrix


def multiSqrt(vectorIn: numpy.ndarray):  # 方根法
    length = vectorIn.size
    multiNum = 1
    for i in vectorIn:
        multiNum = multiNum * i
    return numpy.power(multiNum, 1 / length)


def GRUBBS(matrixIn: numpy.ndarray,alpha:float):
    length = matrixIn.shape[0]
    if length<3:
        raise ValueError("至少需要三个数值")
    meanValue = numpy.mean(matrixIn)
    stdValue = numpy.std(matrixIn,ddof=1)
    if stdValue == 0:
        stdValue = 1
    candidateIndex = numpy.argmax(numpy.abs(matrixIn - meanValue))
    candidate = matrixIn[candidateIndex]
    G = abs(candidate-meanValue)/stdValue
    tCritical = stats.t.ppf(1 - alpha / (2 * length), length - 2)
    GCritical = ((length - 1) / numpy.sqrt(length)) * numpy.sqrt(tCritical ** 2 / (length - 2 + tCritical ** 2))
    if G > GCritical:
        matrixIn[candidateIndex] = meanValue
    return matrixIn






