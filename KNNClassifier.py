import nltk
from glob import glob
import os.path
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

lemmatizer = WordNetLemmatizer()
vocabulary = []
matrix = []

#lê todos os textos de avaliação e adiciona no arquivo vocabulary.txt todas as palavras dos textos de avaliação
def generateVocabulary():
    i = 1
    for filepath in glob('pos\\**'):
        arq = open(filepath,'r')
        text = arq.read()
        text = nltk.word_tokenize(text)
        text = [word.lower() for word in text if word.isalpha()]

        for token in text:
            if token not in vocabulary:
                vocabulary.append(token + " ")
        print('loading... %f%%' %(i*100/40))
        i += 1
    for filepath in glob('neg\\**'):
        arq = open(filepath, 'r')
        text = arq.read()
        text = nltk.word_tokenize(text)
        text = [word.lower() for word in text if word.isalpha()]

        for token in text:
            if token not in vocabulary:
                vocabulary.append(token + " ")
        print('loading... %f%%' % (i * 100 / 40))
        i += 1
    arq = open('vocabulary.txt', 'w')
    for word in vocabulary:
        arq.write(word)

#processa o vocabulário, gera os tokens, lemas e filtra o texto, retirando as stopwords
def toProcess(vocabulary):
    vocabulary = word_tokenize(vocabulary)
    tokens = [lemmatizer.lemmatize(word) for word in vocabulary]
    arq = open('stopwords.txt', 'r')
    stopWords = arq.read()
    stopWords = word_tokenize(stopWords)
    wordsFiltered = []

    for w in tokens:
        if w not in stopWords:
            wordsFiltered.append(w)

    return wordsFiltered

#gera a matriz de classificação, onde cada texto vira um array do tipo
#texto1 = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
#onde cada indice 'i' do array texto1 indica se o text possui a palavra 'i' do vocabulário
#e o ultimo índice indica se o texto é positivo (1) ou negativo (0)
#a matriz gerada não é nada mais que uma lista de listas com todos os arrays
#matriz = [texto1, texto2, texto3, ...]
def generateMatrix(vocabulary):
    i = 0
    for filepath in glob('pos20\\**'):
        arq = open(filepath,'r')
        text = arq.read()
        text = nltk.word_tokenize(text)
        text = [lemmatizer.lemmatize(word) for word in text]
        text = [word.lower() for word in text if word.isalpha()]

        textLine = []

        for word in vocabulary:
            if word in text:
                textLine.append(1)
            else:
                textLine.append(0)
        textLine.append(1)
        matrix.append(textLine)
        print('loading... %f%%' % (i * 100 / 40))
        i += 1

    for filepath in glob('neg20\\**'):
        arq = open(filepath,'r')
        text = arq.read()
        text = nltk.word_tokenize(text)
        text = [lemmatizer.lemmatize(word) for word in text]
        text = [word.lower() for word in text if word.isalpha()]

        textLine = []

        for word in vocabulary:
            if word in text:
                textLine.append(1)
            else:
                textLine.append(0)
        textLine.append(0)
        matrix.append(textLine)
        print('loading... %f%%' % (i * 100 / 40))
        i += 1

def printMatrixToFile(filepath, matrix):
    f = open(filepath, 'w')
    for i in matrix:
        for j in i:
            f.write('%d ' % j)
        f.write('\n')

#calcula a similaridade baseada na quantidade de palavras que as duas linhas (textos) tem em comum
def getSimilarity(lineA, lineB, k):
    n = lineA.__len__()
    similarity = 0
    for i in range (0,n-1):
        if lineA[i] == 1 and lineB == 1:
            similarity += 1

    return similarity

def main():
    if os.path.exists('vocabulary.txt'):
        f = open('vocabulary.txt', 'r')
        vocabulary = f.read()
        vocabulary = toProcess(vocabulary)
        generateMatrix(vocabulary)

        test = ""
        k = input('digite o k a ser utilizado: ')
        print(knn(matrix,test,k))

    else:
        generateVocabulary()
        f = open('vocabulary40.txt', 'r')
        print(f.read)


if __name__ == "__main__":
    main()