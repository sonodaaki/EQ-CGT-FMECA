import numpy


def posterioriProcess(inputMatrix:numpy.ndarray):
    matrixShape = numpy.shape(inputMatrix)
    outputMatrix = createMatrix(matrixShape)
    for r in range(matrixShape[0]):
        outputMatrix[r] = EuclideanDistanceMatrix(inputMatrix[r])
    outputMatrix = sumNorm(outputMatrix,1)
    return outputMatrix


def GRUBBSCheck(matrixIn:numpy.ndarray):
    for j in range(matrixIn.shape[1]):
        for k in range(matrixIn.shape[2]):
            for l in range(matrixIn.shape[3]):
                matrixIn[:,j,k,l] = GRUBBS(matrixIn[:,j,k,l],0.05)
    return matrixIn


def originalNormalizedRPN(matrixS:numpy.ndarray,matrixO:numpy.ndarray,matrixD:numpy.ndarray):
    return 0.34*numpy.average(matrixS,axis=1) * 0.33*numpy.average(matrixO,axis=1) * 0.33* numpy.average(matrixD,axis=1)


def originalRPN(matrixS:numpy.ndarray,matrixO:numpy.ndarray,matrixD:numpy.ndarray):
    return numpy.average(matrixS,axis=1) * numpy.average(matrixO,axis=1) * numpy.average(matrixD,axis=1)


class FAHP:
    def __init__(self):
        self.__expertNumber = 0
        self.__processMatrix = []

    def TFNInput(self,matrixIn:list):
        self.__processMatrix.append(matrixIn)
        self.__expertNumber += 1

    def process(self):
        try:
            npMatrix = GRUBBSCheck(numpy.array(self.__processMatrix))
            averageMatrix = numpy.average(npMatrix,axis=0)
            # print(f"FM:{averageMatrix}")
            if averageMatrix.ndim == 3:
                pass
            else:
                raise
            dispersionMat = dispersionMatrix(averageMatrix)
            mediumMat = averageMatrix[:,:,1]
            quantifiedMat = mediumMat @ dispersionMat
            for colNum in range(quantifiedMat.shape[1]):
                diagnalNumber = quantifiedMat[colNum][colNum]
                for rowNum in range(quantifiedMat.shape[0]):
                    if rowNum == colNum:
                        quantifiedMat[rowNum][colNum] = 1
                    else:
                        quantifiedMat[rowNum][colNum] = quantifiedMat[rowNum][colNum]/diagnalNumber
            weightOutput = numpy.array([])
            for i in quantifiedMat:
                weightOutput = numpy.append(weightOutput,values=multiSqrt(i))
            # print(f"wi = {weightOutput}")
            return sumNorm(weightOutput,dim=0)
        except Exception as e:
            return e


class CRITIC:
    def __init__(self, matrixIn:numpy.ndarray):
        self.__inputMatrix = matrixIn
        self.__intensity = self.__contrastIntensity()
        self.__conflict = self.__conflictDegree()

    def __contrastIntensity(self):
        normMat = zscore(self.__inputMatrix)
        avgNormMat = numpy.average(normMat,axis=0)
        stdMat = numpy.std(avgNormMat,axis=0)
        return stdMat/avgNormMat

    def __conflictDegree(self):
        colNum = self.__inputMatrix.shape[1]
        Rm = []
        for i in range(colNum):
            tempConv = 1
            for j in range(colNum):
                if i == j:
                    continue
                else:
                    tempConv = tempConv * (1-abs(numpy.corrcoef(self.__inputMatrix[:,i],self.__inputMatrix[:,j])[0,1]))
            Rm.append(tempConv)
        return numpy.array(Rm)

    def process(self):
        return sumNorm((self.__intensity * self.__conflict),0)


class MEREC:
    def __init__(self, matrixIn: numpy.ndarray):
        self.__inputMatrix = matrixIn
        self.overall = self.__overallPerformance()
        self.removal = self.__removalPerformance()

    def __overallPerformance(self):
        normMat = maxNorm(self.__inputMatrix,0)
        logn = numpy.sum(numpy.abs(numpy.log(normMat)),axis= 1)
        Si = numpy.log(1 + (1/(self.__inputMatrix.shape[1]))* logn)
        return Si

    def __removalPerformance(self):
        normMat = maxNorm(self.__inputMatrix,0)
        outputMat = numpy.zeros(shape=(normMat.shape[1],normMat.shape[0]))
        for i in range(outputMat.shape[0]):
            logn = numpy.sum(numpy.abs(numpy.log(normMat)), axis=1) - numpy.abs(numpy.log(normMat))[:,i]
            Sij = numpy.log(1 + (1 / (self.__inputMatrix.shape[1])) * logn)
            outputMat[i] = Sij
        return outputMat

    def process(self):
        Ej = numpy.sum(numpy.abs(self.removal - self.overall),axis=1)
        # print(f"Em = {Ej}")
        return sumNorm(Ej,0)


class ENTROPY:
    def __init__(self, matrixIn: numpy.ndarray, indicator_type='cost'):
        self.__inputMatrix = numpy.asarray(matrixIn, dtype=float)
        self.m, self.n = self.__inputMatrix.shape

        if indicator_type is None:
            self.indicator_type = ['benefit'] * self.n
        else:
            self.indicator_type = indicator_type

        self.__normalizedMatrix = None
        self.__weights = None

    def __normalize(self):
        X = self.__inputMatrix.copy()
        for j in range(self.n):
            if self.indicator_type[j] == 'cost':
                # 成本型 → 转化为效益型
                X[:, j] = X[:, j].max() - X[:, j]
        # 极差标准化
        eps = 1e-12
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + eps)
        self.__normalizedMatrix = X_norm
        return X_norm

    def __calculate_weights(self):
        X_norm = self.__normalize()
        eps = 1e-12

        P = X_norm / (X_norm.sum(axis=0) + eps)
        k = 1.0 / numpy.log(self.m + eps)
        E = -k * numpy.sum(P * numpy.log(P + eps), axis=0)

        d = 1 - E
        if numpy.all(d < eps):
            self.__weights = numpy.ones(self.n) / self.n
        else:
            self.__weights = d / (d.sum() + eps)

        return self.__weights

    def process(self):
        return self.__calculate_weights()

    def get_normalized_matrix(self):
        if self.__normalizedMatrix is None:
            self.__normalize()
        return self.__normalizedMatrix

    def get_weights(self):
        if self.__weights is None:
            return self.process()
        return self.__weights


class TOPSIS:
    def __init__(self, matrixIn: numpy.ndarray, weights=None, benefit_attributes=None):
        self.__inputMatrix = matrixIn.astype(float)
        self.n_alternatives, self.n_criteria = matrixIn.shape

        # 设置权重
        if weights is None:
            self.__weights = numpy.ones(self.n_criteria) / self.n_criteria
        else:
            if len(weights) != self.n_criteria:
                raise ValueError(f"权重数量({len(weights)})必须与指标数量({self.n_criteria})一致")
            self.__weights = numpy.array(weights) / numpy.sum(weights)  # 归一化权重

        if benefit_attributes is None:
            self.__benefit_attributes = numpy.ones(self.n_criteria, dtype=bool)
        else:
            if len(benefit_attributes) != self.n_criteria:
                raise ValueError(f"指标类型数量({len(benefit_attributes)})必须与指标数量({self.n_criteria})一致")
            self.__benefit_attributes = numpy.array(benefit_attributes, dtype=bool)

    def __vector_normalization(self):
        """向量归一化"""
        squared_sum = numpy.sqrt(numpy.sum(self.__inputMatrix ** 2, axis=0))
        return self.__inputMatrix / squared_sum

    def __minmax_normalization(self):
        """最小-最大归一化"""
        mins = numpy.min(self.__inputMatrix, axis=0)
        maxs = numpy.max(self.__inputMatrix, axis=0)
        range_vals = maxs - mins
        range_vals[range_vals == 0] = 1  # 避免除以0
        return (self.__inputMatrix - mins) / range_vals

    def __zscore_normalization(self):
        """Z-score归一化"""
        means = numpy.mean(self.__inputMatrix, axis=0)
        stds = numpy.std(self.__inputMatrix, axis=0)
        stds[stds == 0] = 1  # 避免除以0
        return (self.__inputMatrix - means) / stds

    def __linear_normalization(self):
        """线性比例归一化（适用于TOPSIS的常用方法）"""
        norm_matrix = numpy.zeros_like(self.__inputMatrix)

        for j in range(self.n_criteria):
            if self.__benefit_attributes[j]:
                # 效益型指标：x/max
                max_val = numpy.max(self.__inputMatrix[:, j])
                if max_val == 0:
                    norm_matrix[:, j] = 0
                else:
                    norm_matrix[:, j] = self.__inputMatrix[:, j] / max_val
            else:
                # 成本型指标：min/x
                min_val = numpy.min(self.__inputMatrix[:, j])
                if min_val == 0:
                    norm_matrix[:, j] = 0
                else:
                    norm_matrix[:, j] = min_val / self.__inputMatrix[:, j]

        return norm_matrix

    def __create_weighted_matrix(self, norm_matrix):
        """创建加权规范化矩阵"""
        return norm_matrix * self.__weights

    def __determine_ideal_solutions(self, weighted_matrix):
        """确定正理想解和负理想解"""
        # 初始化理想解
        positive_ideal = numpy.zeros(self.n_criteria)
        negative_ideal = numpy.zeros(self.n_criteria)

        for j in range(self.n_criteria):
            if self.__benefit_attributes[j]:
                # 效益型指标：正理想取最大值，负理想取最小值
                positive_ideal[j] = numpy.max(weighted_matrix[:, j])
                negative_ideal[j] = numpy.min(weighted_matrix[:, j])
            else:
                # 成本型指标：正理想取最小值，负理想取最大值
                positive_ideal[j] = numpy.min(weighted_matrix[:, j])
                negative_ideal[j] = numpy.max(weighted_matrix[:, j])

        return positive_ideal, negative_ideal

    def __calculate_distances(self, weighted_matrix, positive_ideal, negative_ideal):
        d_plus = numpy.sqrt(numpy.sum((weighted_matrix - positive_ideal) ** 2, axis=1))
        d_minus = numpy.sqrt(numpy.sum((weighted_matrix - negative_ideal) ** 2, axis=1))

        return d_plus, d_minus

    def __calculate_closeness(self, d_plus, d_minus):
        """计算相对贴近度"""
        # 避免除以0
        denominator = d_plus + d_minus
        denominator[denominator == 0] = 1e-10

        closeness = d_minus / denominator
        return closeness

    def process(self, normalization_method='linear'):
        # 1. 归一化决策矩阵
        if normalization_method == 'vector':
            norm_matrix = self.__vector_normalization()
        elif normalization_method == 'minmax':
            norm_matrix = self.__minmax_normalization()
        elif normalization_method == 'zscore':
            norm_matrix = self.__zscore_normalization()
        elif normalization_method == 'linear':
            norm_matrix = self.__linear_normalization()
        else:
            raise ValueError(f"不支持的归一化方法: {normalization_method}")

        # 2. 构造加权规范化矩阵
        weighted_matrix = self.__create_weighted_matrix(norm_matrix)

        # 3. 确定正负理想解
        positive_ideal, negative_ideal = self.__determine_ideal_solutions(weighted_matrix)

        # 4. 计算距离
        d_plus, d_minus = self.__calculate_distances(weighted_matrix, positive_ideal, negative_ideal)

        # 5. 计算相对贴近度
        closeness = self.__calculate_closeness(d_plus, d_minus)

        # 6. 排序
        from scipy.stats import rankdata
        rank = rankdata(-closeness, method='min')  # 贴近度越大越好

        results = {
            'normalized_matrix': norm_matrix,
            'weighted_matrix': weighted_matrix,
            'positive_ideal': positive_ideal,
            'negative_ideal': negative_ideal,
            'distance_positive': d_plus,
            'distance_negative': d_minus,
            'closeness': closeness,
            'rank': rank,
            'best_alternative': numpy.argmax(closeness),  # 最佳方案索引
            'weights': self.__weights.copy()
        }

        return results

    def process_with_custom_weights(self, weights, normalization_method='linear'):
        # 临时保存原始权重
        original_weights = self.__weights.copy()

        # 设置新权重
        if len(weights) != self.n_criteria:
            raise ValueError(f"权重数量({len(weights)})必须与指标数量({self.n_criteria})一致")
        self.__weights = numpy.array(weights) / numpy.sum(weights)  # 归一化权重

        # 执行TOPSIS
        results = self.process(normalization_method)

        # 恢复原始权重
        self.__weights = original_weights

        return results


def gameTheory(vectorIn1:numpy.ndarray,vectorIn2:numpy.ndarray):
    w1 = vectorIn1
    w2 = vectorIn2
    w1_T = w1.T
    w2_T = w2.T
    AA = w1.dot(w1_T).reshape(1)
    BB = w1.dot(w2_T).reshape(1)
    CC = w2.dot(w1_T).reshape(1)
    DD = w2.dot(w2_T).reshape(1)

    mm = numpy.array([[AA, BB], [CC, DD]]).reshape(2, 2)
    Y = numpy.concatenate((AA, DD)).reshape(2, 1)
    re = numpy.linalg.solve(mm, Y)
    d1 = re[0] / (re.sum())
    d2 = re[1] / (re.sum())
    w = w1 * d1 + w2 * d2
    # print(f"epsilon:{[d1,d2]}")
    return w


def fuse_weights(w_s, w_o, method='MDI_reverse', alpha=0.5, eps=1e-12):

    w_s = numpy.asarray(w_s, dtype=float).flatten()
    w_o = numpy.asarray(w_o, dtype=float).flatten()
    if w_s.shape != w_o.shape:
        raise ValueError("w_s and w_o must have the same shape.")
    n = w_s.size
    # 保证非负
    w_s = numpy.maximum(w_s, 0.0)
    w_o = numpy.maximum(w_o, 0.0)
    # 防止全零
    if w_s.sum() == 0:
        w_s = numpy.ones_like(w_s) / n
    else:
        w_s = w_s / w_s.sum()
    if w_o.sum() == 0:
        w_o = numpy.ones_like(w_o) / n
    else:
        w_o = w_o / w_o.sum()
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    if method == 'MDI_forward' or method == 'geometric':
        # w_i ∝ w_s_i^alpha * w_o_i^(1-alpha)
        # 为避免 0^power 带来的问题，先加 eps
        base_s = numpy.maximum(w_s, eps)
        base_o = numpy.maximum(w_o, eps)
        w_unnorm = (base_s ** alpha) * (base_o ** (1.0 - alpha))
        w = w_unnorm / w_unnorm.sum()
        return w

    if method == 'MDI_reverse' or method == 'linear' or method == 'least_squares':
        # 证明：最小化 alpha D(w_s||w) + (1-alpha) D(w_o||w) 等价于最大化 sum (alpha w_s + (1-alpha) w_o) log w
        # 在约束 sum w = 1, w>=0 下得到闭式解：
        # w_i ∝ alpha * w_s_i + (1-alpha) * w_o_i
        w_unnorm = alpha * w_s + (1.0 - alpha) * w_o
        # 若最小二乘形式，求导也可得到同样的加权平均解（带约束的最小平方）
        # 直接归一化
        w_unnorm = numpy.maximum(w_unnorm, eps)  # 防止为0
        w = w_unnorm / w_unnorm.sum()
        return w

    raise ValueError(
        f"Unknown method '{method}'. Supported: 'MDI_forward','MDI_reverse','linear','least_squares','geometric'.")




