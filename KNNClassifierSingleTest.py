import nltk
from glob import glob
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle
from GetSimilarities import *
import math

matrix = []
lemmatizer = WordNetLemmatizer()
dictionary = []
vocabulary = {}
#processa o vocabulário, gera os tokens, lemas e filtra o texto, retirando as stopwords
def getTF(word, vocabulary):
    return word[1]/vocabulary[0][1]


def getIDF(word, idf):
    if idf[word[0]] == 0:
        return 0
    return math.log2(1000/idf[word[0]])


def toProcess(vocabulary,n):
    with open('idf', 'rb') as fp:
        idf = pickle.load(fp)

    print('processing vocabulary')
    arq = open('stopwords.txt', 'r')
    stopWords = arq.read()
    stopWords = word_tokenize(stopWords)

    filteredVocabulary = []
    for w in vocabulary:
        if w not in stopWords:
            filteredVocabulary.append([w,vocabulary[w]])

    filteredVocabulary.sort(key=lambda x: x[1], reverse=True)
    dictionary = []
    for word in filteredVocabulary:
        tfidf = getTF(word, filteredVocabulary) * getIDF(word,idf)
        dictionary.append([word[0],tfidf])


    dictionary.sort(key=lambda x: x[1], reverse=True)

    dictionary = dictionary[0:n]

    arq.close()
    return dictionary

#gera a matriz de classificação, onde cada texto vira um array do tipo
#texto1 = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
#onde cada indice 'i' do array texto1 indica se o text possui a palavra 'i' do vocabulário
#e o ultimo índice indica se o texto é positivo (1) ou negativo (0)
#a matriz gerada não é nada mais que uma lista de listas com todos os arrays
#matriz = [texto1, texto2, texto3, ...]
def generateMatrix(dictionary):
    i = 1
    global matrix
    matrix = []
    for filepath in glob('pos\\**'):
        arq = open(filepath,'r')
        text = arq.read()
        text = nltk.word_tokenize(text)
        text = [lemmatizer.lemmatize(word) for word in text]
        text = [word.lower() for word in text if word.isalpha()]

        textLine = []

        for word in dictionary:
            if word[0] in text:
                textLine.append(1)
            else:
                textLine.append(0)
        textLine.append(1)
        matrix.append(textLine)
        if i%200 == 0:
            print('loading matrix... %f%%' % (i * 100 / 1000))
        i += 1

    for filepath in glob('neg\\**'):
        arq = open(filepath,'r')
        text = arq.read()
        text = nltk.word_tokenize(text)
        text = [lemmatizer.lemmatize(word) for word in text]
        text = [word.lower() for word in text if word.isalpha()]

        textLine = []

        for word in dictionary:
            if word[0] in text:
                textLine.append(1)
            else:
                textLine.append(0)
        textLine.append(0)
        matrix.append(textLine)
        if i%200 == 0:
            print('loading matrix... %f%%' % (i * 100 / 1000))
        i += 1

def printMatrix(matrix):
    for i in matrix:
        for j in i:
            print(j, end='')
        print()

def getBonusSimilarity(lineA, lineB):
    n = lineA.__len__()
    similarity = 0
    for i in range(0, n - 1):
        df = vocabulary[dictionary[i][0]]
        if lineA[i] == 1 and lineB[i] == 1:
            similarity = similarity + 1 + (1/df)
    return similarity

def defineSentiment(matrix,distanceVector):
    positive = 0
    negative = 0
    while positive == negative:
        positive = 0
        negative = 0
        for i in range(0, distanceVector.__len__()):
            j = distanceVector[i][0]
            n = matrix[0].__len__()
            m = matrix.__len__()
            if matrix[j][n - 1] == 1:
                positive += 1
            else:
                negative += 1
        distanceVector.pop()

    if positive > negative:
        return 1
    else:
        return 0

def knn(dictionary, k):
    print(dictionary.__len__())
    generateMatrix(dictionary)
    print(matrix[0].__len__())
    accuracy = []
    # K-folder:
    kf = KFold(n_splits=10)
    for train, test in kf.split(matrix):
        # gera dois vetores que armazenam as posições do vetor MATRIX que serão utilizadas
        # para teste e treinamento
        # Primeiro realizar o treinamento e depois o teste
        # Calcular Métricas:  Precision, Recall, F-measure e Accuracy
        # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics (tem na biblioteca do sklearn)

        result = []
        print('defining sentiment...')
        for i in test:
            distanceVector = []
            for j in train:
                distanceVector.append([j,jaccardSimilarity(matrix[i],matrix[j])])

            distanceVector.sort(key=lambda x: x[1], reverse = True)

            while distanceVector.__len__() > k:
                distanceVector.pop()
            result.append([matrix[i][dictionary.__len__()], defineSentiment(matrix,distanceVector)])
        acc = 0
        for i in result:
            if i[0] == i[1]:
                acc += 1
        accuracy.append(acc / result.__len__() * 100)

    return accuracy


def main():
    print('loading vocabulary...')
    with open('vocabulary', 'rb') as fp:
        global vocabulary
        vocabulary = pickle.load(fp)
    print('vocabulary has been loaded.')

    #k = input('digite o k a ser utilizado: ')
    #n = input('digite o tamanho do vocabulario: ')

    dictionary = toProcess(vocabulary, 2500)
    accuracy = knn(dictionary,5)
    print(sum(accuracy)/10)

if __name__ == "__main__":
    main()