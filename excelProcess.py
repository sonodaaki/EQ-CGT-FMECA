import numpy
import xlrd
import xlwt


def excelInput(path:str,matrixIn:numpy.ndarray):
    excelData = xlrd.open_workbook(path).sheets()[0]
    rowNum = matrixIn.shape[0]
    colNum = matrixIn.shape[1]
    for r in range(rowNum):
        for c in range(colNum):
            matrixIn[r][c] = excelData.cell_value(r,c)
    return matrixIn


def excelOutput(path:str,matrixIn):
    if isinstance(matrixIn,numpy.ndarray):
        workBook = xlwt.Workbook(encoding='utf-8')
        sheet = workBook.add_sheet("result")
        dims = matrixIn.ndim
        if dims == 1:
            for i in range(matrixIn.shape[0]):
                sheet.write(i,0,matrixIn[i])
        else:
            for i in range(matrixIn.shape[0]):
                for j in range(matrixIn.shape[1]):
                    sheet.write(i, j, matrixIn[i][j])

        workBook.save(path)
    else:
        workBook = xlwt.Workbook(encoding='utf-8')
        sheet = workBook.add_sheet("result")
        for i in range(len(matrixIn)):
            for j in range(len(matrixIn[i])):
                sheet.write(i, j, matrixIn[i][j])

        workBook.save(path)


def excelRead(path:str):
    excelData = xlrd.open_workbook(path).sheets()[0]
    rowNum = excelData.nrows
    colNum = excelData.ncols
    matrixIn = []
    for r in range(rowNum):
        tempList = []
        for c in range(colNum):
            tempList.append(excelData.cell_value(r,c))
        matrixIn.append(tempList)
    return matrixIn
