def jaccardSimilarity(lineA, lineB):
    n = lineA.__len__()
    n -= 1
    m1 = 0
    m2 = 0

    for i in range (0,n):
        if lineA[i] == 1 and lineB[i] == 1:
            m1 += 1
        elif lineA[i] == 1 and lineB[i] == 0:
            m2 += 1
        elif lineA[i] == 0 and lineB[i] == 1:
            m2 += 1
    similarity = m1/(m1+m2+m2)
    return similarity

def getSimilarity(lineA, lineB):
    n = lineA.__len__()
    similarity = 0
    for i in range (0,n-1):
        #df = vocabulary[dictionary[i][0]]
        if lineA[i] == 1 and lineB[i] == 1:
            similarity = similarity + 1 #+ (1/df)
    return similarity

def diceSimilarity(lineA, lineB):
    sizeA = len(lineA)-1
    sizeB = len(lineB)-1
    tp = 0
    f = 0
    for i in range (lineA.__len__()-1):
        if lineA[i] == 1 and lineB[i] == 1:
            tp += 1
        elif lineA[i] == 1 and lineB[i] == 0:
            f += 1
        elif lineA[i] == 0 and lineB[i] == 1:
            f += 1
    similarity = (2*tp)/(2*tp+f)
    return similarity