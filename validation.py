import numpy

from excelProcess import *
from algorithm import *
from matrixProcess import *


def searchKeywords(path:str,keywords: str):
    inputData = excelRead(path)
    output = []
    for i in inputData:
        for j in i:
            if keywords in j:
                output.append(i)
                break
    return output


class validationProcess:
    def __init__(self,expertNum:int,fmNum:int,matrixIn:list,X_S:numpy.ndarray,X_O:numpy.ndarray,X_D:numpy.ndarray,NAIn:numpy.ndarray):
        self.expertNum = expertNum
        self.fmNum = fmNum
        self.pairwiseMat = matrixIn
        self.severity = X_S
        self.occurrence = X_O
        self.detectability = X_D
        self.naturalAttr = NAIn

    def singleFMECA(self):
        result = originalNormalizedRPN(self.severity,self.occurrence,self.detectability)
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def singleFAHP(self):
        avgS = numpy.average(self.severity,axis=1)
        avgO = numpy.average(self.occurrence, axis=1)
        avgD = numpy.average(self.detectability, axis=1)
        FAHPLayer = FAHP()
        for i in self.pairwiseMat:
            FAHPLayer.TFNInput(i)
        W_FAHP = FAHPLayer.process()
        result = W_FAHP[0]*avgS*W_FAHP[1]*avgO*W_FAHP[2]*avgD
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def singleMEREC(self):
        avgS = numpy.average(self.severity, axis=1)
        avgO = numpy.average(self.occurrence, axis=1)
        avgD = numpy.average(self.detectability, axis=1)
        processedData = numpy.array([avgS, avgO, avgD]).T
        W_MEREC = MEREC(processedData).process()
        result = W_MEREC[0]*avgS*W_MEREC[1]*avgO*W_MEREC[2]*avgD
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def singleCRITIC(self):
        avgS = numpy.average(self.severity, axis=1)
        avgO = numpy.average(self.occurrence, axis=1)
        avgD = numpy.average(self.detectability, axis=1)
        processedData = numpy.array([avgS, avgO, avgD]).T
        W_CRITIC = CRITIC(processedData).process()
        result = W_CRITIC[0]*avgS*W_CRITIC[1]*avgO*W_CRITIC[2]*avgD
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def singleENTROPY(self):
        avgS = numpy.average(self.severity, axis=1)
        avgO = numpy.average(self.occurrence, axis=1)
        avgD = numpy.average(self.detectability, axis=1)
        processedData = numpy.array([avgS, avgO, avgD]).T
        W_ENTROPY = ENTROPY(processedData).process()
        result = W_ENTROPY[0]*avgS*W_ENTROPY[1]*avgO*W_ENTROPY[2]*avgD
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def ENTROPY_TOPSIS(self):
        avgS = numpy.average(self.severity, axis=1)
        avgO = numpy.average(self.occurrence, axis=1)
        avgD = numpy.average(self.detectability, axis=1)
        processedData = numpy.array([avgS, avgO, avgD]).T
        W_ENTROPY = ENTROPY(processedData).process()
        benefit_attributes = [True, True, True]
        TOPSISR = TOPSIS(processedData, W_ENTROPY, benefit_attributes).process()

        return TOPSISR['normalized_matrix'],TOPSISR['rank']

    def FAHP_TOPSIS(self):
        avgS = numpy.average(self.severity, axis=1)
        avgO = numpy.average(self.occurrence, axis=1)
        avgD = numpy.average(self.detectability, axis=1)
        processedData = numpy.array([avgS, avgO, avgD]).T
        FAHPLayer = FAHP()
        for i in self.pairwiseMat:
            FAHPLayer.TFNInput(i)
        W_FAHP = FAHPLayer.process()
        benefit_attributes = [True, True, True]
        TOPSISR = TOPSIS(processedData, W_FAHP, benefit_attributes).process()

        return TOPSISR['normalized_matrix'],TOPSISR['rank']

    def withoutExpertQualification(self):
        avgS = numpy.average(self.severity, axis=1)
        avgO = numpy.average(self.occurrence, axis=1)
        avgD = numpy.average(self.detectability, axis=1)
        processedData = numpy.array([avgS, avgO, avgD]).T
        FAHPLayer = FAHP()
        for i in self.pairwiseMat:
            FAHPLayer.TFNInput(i)
        W_FAHP = FAHPLayer.process()
        W_MEREC = MEREC(processedData).process()
        W_final = gameTheory(W_FAHP, W_MEREC)
        result = processedData @ W_final
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def integratedApproach(self):
        NA = maxNorm(self.naturalAttr, 0)
        w_NA = numpy.array([0.5, 0.5]).T

        # Process provided data
        w_SO = NA @ w_NA
        w_S = sumNorm(w_SO, 0)
        SW_S = createMatrix([self.fmNum, self.expertNum])
        SW_O = createMatrix([self.fmNum, self.expertNum])
        SW_D = createMatrix([self.fmNum, self.expertNum])
        for SW in [SW_S, SW_O, SW_D]:
            for i in range(self.fmNum):
                SW[i] = w_S
        OW_S = posterioriProcess(self.severity)
        OW_O = posterioriProcess(self.occurrence)
        OW_D = posterioriProcess(self.detectability)

        WP_S = 0.5 * SW_S + 0.5 * OW_S
        WP_O = 0.5 * SW_O + 0.5 * OW_O
        WP_D = 0.5 * SW_D + 0.5 * OW_D

        aggrSN = numpy.sum(numpy.multiply(self.severity, WP_S), axis=1)
        aggrON = numpy.sum(numpy.multiply(self.occurrence, WP_O), axis=1)
        aggrDN = numpy.sum(numpy.multiply(self.detectability, WP_D), axis=1)

        # weight determination
        processedData = numpy.array([aggrSN, aggrON, aggrDN]).T
        FAHPLayer = FAHP()
        for i in self.pairwiseMat:
            FAHPLayer.TFNInput(i)
        W_FAHP = FAHPLayer.process()
        W_MEREC = MEREC(processedData).process()
        W_final = gameTheory(W_FAHP, W_MEREC)
        result = processedData @ W_final
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def integratedApproachLinear(self):
        NA = maxNorm(self.naturalAttr, 0)
        w_NA = numpy.array([0.5, 0.5]).T

        # Process provided data
        w_SO = NA @ w_NA
        w_S = sumNorm(w_SO, 0)
        SW_S = createMatrix([self.fmNum, self.expertNum])
        SW_O = createMatrix([self.fmNum, self.expertNum])
        SW_D = createMatrix([self.fmNum, self.expertNum])
        for SW in [SW_S, SW_O, SW_D]:
            for i in range(self.fmNum):
                SW[i] = w_S
        OW_S = posterioriProcess(self.severity)
        OW_O = posterioriProcess(self.occurrence)
        OW_D = posterioriProcess(self.detectability)

        WP_S = 0.5 * SW_S + 0.5 * OW_S
        WP_O = 0.5 * SW_O + 0.5 * OW_O
        WP_D = 0.5 * SW_D + 0.5 * OW_D

        aggrSN = numpy.sum(numpy.multiply(self.severity, WP_S), axis=1)
        aggrON = numpy.sum(numpy.multiply(self.occurrence, WP_O), axis=1)
        aggrDN = numpy.sum(numpy.multiply(self.detectability, WP_D), axis=1)

        # weight determination
        processedData = numpy.array([aggrSN, aggrON, aggrDN]).T
        FAHPLayer = FAHP()
        for i in self.pairwiseMat:
            FAHPLayer.TFNInput(i)
        W_FAHP = FAHPLayer.process()
        W_MEREC = MEREC(processedData).process()
        W_final = fuse_weights(W_FAHP, W_MEREC,"linear")
        result = processedData @ W_final
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def integratedApproachLeastSquare(self):
        NA = maxNorm(self.naturalAttr, 0)
        w_NA = numpy.array([0.5, 0.5]).T

        # Process provided data
        w_SO = NA @ w_NA
        w_S = sumNorm(w_SO, 0)
        SW_S = createMatrix([self.fmNum, self.expertNum])
        SW_O = createMatrix([self.fmNum, self.expertNum])
        SW_D = createMatrix([self.fmNum, self.expertNum])
        for SW in [SW_S, SW_O, SW_D]:
            for i in range(self.fmNum):
                SW[i] = w_S
        OW_S = posterioriProcess(self.severity)
        OW_O = posterioriProcess(self.occurrence)
        OW_D = posterioriProcess(self.detectability)

        WP_S = 0.5 * SW_S + 0.5 * OW_S
        WP_O = 0.5 * SW_O + 0.5 * OW_O
        WP_D = 0.5 * SW_D + 0.5 * OW_D

        aggrSN = numpy.sum(numpy.multiply(self.severity, WP_S), axis=1)
        aggrON = numpy.sum(numpy.multiply(self.occurrence, WP_O), axis=1)
        aggrDN = numpy.sum(numpy.multiply(self.detectability, WP_D), axis=1)

        # weight determination
        processedData = numpy.array([aggrSN, aggrON, aggrDN]).T
        FAHPLayer = FAHP()
        for i in self.pairwiseMat:
            FAHPLayer.TFNInput(i)
        W_FAHP = FAHPLayer.process()
        W_MEREC = MEREC(processedData).process()
        W_final = fuse_weights(W_FAHP, W_MEREC,"least_squares")
        result = processedData @ W_final
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def integratedApproachGeometric(self):
        NA = maxNorm(self.naturalAttr, 0)
        w_NA = numpy.array([0.5, 0.5]).T

        # Process provided data
        w_SO = NA @ w_NA
        w_S = sumNorm(w_SO, 0)
        SW_S = createMatrix([self.fmNum, self.expertNum])
        SW_O = createMatrix([self.fmNum, self.expertNum])
        SW_D = createMatrix([self.fmNum, self.expertNum])
        for SW in [SW_S, SW_O, SW_D]:
            for i in range(self.fmNum):
                SW[i] = w_S
        OW_S = posterioriProcess(self.severity)
        OW_O = posterioriProcess(self.occurrence)
        OW_D = posterioriProcess(self.detectability)

        WP_S = 0.5 * SW_S + 0.5 * OW_S
        WP_O = 0.5 * SW_O + 0.5 * OW_O
        WP_D = 0.5 * SW_D + 0.5 * OW_D

        aggrSN = numpy.sum(numpy.multiply(self.severity, WP_S), axis=1)
        aggrON = numpy.sum(numpy.multiply(self.occurrence, WP_O), axis=1)
        aggrDN = numpy.sum(numpy.multiply(self.detectability, WP_D), axis=1)

        # weight determination
        processedData = numpy.array([aggrSN, aggrON, aggrDN]).T
        FAHPLayer = FAHP()
        for i in self.pairwiseMat:
            FAHPLayer.TFNInput(i)
        W_FAHP = FAHPLayer.process()
        W_MEREC = MEREC(processedData).process()
        W_final = fuse_weights(W_FAHP, W_MEREC,"geometric")
        result = processedData @ W_final
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def integratedApproachMDI(self):
        NA = maxNorm(self.naturalAttr, 0)
        w_NA = numpy.array([0.5, 0.5]).T

        # Process provided data
        w_SO = NA @ w_NA
        w_S = sumNorm(w_SO, 0)
        SW_S = createMatrix([self.fmNum, self.expertNum])
        SW_O = createMatrix([self.fmNum, self.expertNum])
        SW_D = createMatrix([self.fmNum, self.expertNum])
        for SW in [SW_S, SW_O, SW_D]:
            for i in range(self.fmNum):
                SW[i] = w_S
        OW_S = posterioriProcess(self.severity)
        OW_O = posterioriProcess(self.occurrence)
        OW_D = posterioriProcess(self.detectability)

        WP_S = 0.5 * SW_S + 0.5 * OW_S
        WP_O = 0.5 * SW_O + 0.5 * OW_O
        WP_D = 0.5 * SW_D + 0.5 * OW_D

        aggrSN = numpy.sum(numpy.multiply(self.severity, WP_S), axis=1)
        aggrON = numpy.sum(numpy.multiply(self.occurrence, WP_O), axis=1)
        aggrDN = numpy.sum(numpy.multiply(self.detectability, WP_D), axis=1)

        # weight determination
        processedData = numpy.array([aggrSN, aggrON, aggrDN]).T
        FAHPLayer = FAHP()
        for i in self.pairwiseMat:
            FAHPLayer.TFNInput(i)
        W_FAHP = FAHPLayer.process()
        W_MEREC = MEREC(processedData).process()
        W_final = fuse_weights(W_FAHP, W_MEREC,"MDI_forward")
        result = processedData @ W_final
        ranks = numpy.argsort(-result)+1
        return result,ranks

    def integratedApproachSensitiveAnalysis(self,alpha,delta):

        NA = maxNorm(self.naturalAttr, 0)
        w_NA = numpy.array([alpha, 1-alpha]).T

        # Process provided data
        w_SO = NA @ w_NA
        w_S = sumNorm(w_SO, 0)
        SW_S = createMatrix([self.fmNum, self.expertNum])
        SW_O = createMatrix([self.fmNum, self.expertNum])
        SW_D = createMatrix([self.fmNum, self.expertNum])
        for SW in [SW_S, SW_O, SW_D]:
            for i in range(self.fmNum):
                SW[i] = w_S
        OW_S = posterioriProcess(self.severity)
        OW_O = posterioriProcess(self.occurrence)
        OW_D = posterioriProcess(self.detectability)

        WP_S = delta * SW_S + (1-delta) * OW_S
        WP_O = delta * SW_O + (1-delta) * OW_O
        WP_D = delta * SW_D + (1-delta) * OW_D

        aggrSN = numpy.sum(numpy.multiply(self.severity, WP_S), axis=1)
        aggrON = numpy.sum(numpy.multiply(self.occurrence, WP_O), axis=1)
        aggrDN = numpy.sum(numpy.multiply(self.detectability, WP_D), axis=1)

        # weight determination
        processedData = numpy.array([aggrSN, aggrON, aggrDN]).T
        FAHPLayer = FAHP()
        for i in self.pairwiseMat:
            FAHPLayer.TFNInput(i)
        W_FAHP = FAHPLayer.process()
        W_MEREC = MEREC(processedData).process()
        W_final = gameTheory(W_FAHP, W_MEREC)
        result = processedData @ W_final
        ranks = numpy.argsort(-result)+1
        return result,ranks


if __name__ == '__main__':
    expertNumber = 6
    failureNumber = 31
    XS = excelInput('inputdata/SN.xls', createMatrix([failureNumber, expertNumber]))
    XO = excelInput('inputdata/ON.xls', createMatrix([failureNumber, expertNumber]))
    XD = excelInput('inputdata/DN.xls', createMatrix([failureNumber, expertNumber]))
    NAInput = excelInput('inputdata/NA.xls', createMatrix([expertNumber, 2]))
    TFN1 = [[[1, 1, 1], [1, 3, 7], [1, 3, 7]],
            [[1 / 7, 1 / 3, 1], [1, 1, 1], [1, 3, 7]],
            [[1 / 7, 1 / 3, 1], [1 / 7, 1 / 3, 1], [1, 1, 1]]]
    TFN2 = [[[1, 1, 1], [1 / 5, 1 / 3, 1], [1 / 5, 1 / 3, 1]],
            [[1, 3, 5], [1, 1, 1], [1, 2, 3]],
            [[1, 3, 5], [1 / 3, 1 / 2, 1], [1, 1, 1]]]
    TFN3 = [[[1, 1, 1], [1, 1, 1], [2, 2, 2]],
            [[1, 1, 1], [1, 1, 1], [3, 3, 3]],
            [[1 / 2, 1 / 2, 1 / 2], [1 / 3, 1 / 3, 1 / 3], [1, 1, 1]]]
    TFN4 = [[[1, 1, 1], [1 / 7, 1 / 7, 1 / 7], [1, 1, 1]],
            [[7, 7, 7], [1, 1, 1], [2, 2, 2]],
            [[1, 1, 1], [1 / 2, 1 / 2, 1 / 2], [1, 1, 1]]]
    TFN5 = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    TFN6 = [[[1, 1, 1], [1, 3, 7], [1, 3, 7]],
            [[1 / 7, 1 / 3, 1], [1, 1, 1], [1, 3, 7]],
            [[1 / 7, 1 / 3, 1], [1 / 7, 1 / 3, 1], [1, 1, 1]]]
    TFNMatrix = [TFN1, TFN2, TFN3, TFN4, TFN5, TFN6]
    test = validationProcess(expertNumber,failureNumber,TFNMatrix,XS,XO,XD,NAInput)
    print(test.integratedApproachSensitiveAnalysis(0.5,0.5))
    # print(test.ENTROPY_TOPSIS())
    # print(test.FAHP_TOPSIS())
    # print(test.singleCRITIC())
    # print(test.singleENTROPY())
    # print(test.singleFMECA())
    # print(test.singleFAHP())
    # print(test.singleMEREC())
    # print(test.withoutExpertQualification())
    # print(test.integratedApproach())
    # words = ['整流','TCU','电机','牵引电机','主断','弓','变压','直流']
    # for w in words:
    #     res = test.searchKeywords(w)
    #     excelOutput(f"valFiles/{w}.xls",res)