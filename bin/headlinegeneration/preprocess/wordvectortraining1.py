__author__ = 'Chi'

from time import time
from bin.headlinegeneration.utils.loginfor import loginformation
from gensim.models import word2vec
import re
import multiprocessing
import gensim
import os


def cleandata(rawdata):
    try:
        cleanr = re.compile('<.*>?>')
        cleantext = re.sub(cleanr, ' ', rawdata)
    except:
        pass
        #loginformation("", "debug", "Error occurs while cleaning data: " + rawdata)
        #loginformation("", "info", "Finished clean the data: " + rawdata)
    return cleantext

#生成迭代器，以防止数据太大，内存不够
class GetSentence(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):        #__iter__方法的对象是可迭代的。
        for file in os.listdir(self.dirname):
            filePath = self.dirname + '/' + file
            try:
                for line in open(filePath):
                    sline = line.strip()
                    if sline == "":
                        continue
                    rline = cleandata(sline)
                    yield rline.split(" ")
            except IOError:
                loginformation("", "info", "open file: " + self.dirname + '/' + file + " error ")
            else:
                loginformation("", "info", "finish the file: " + self.dirname + '/' + file)

def wordVectorTrainModel(trainDataPath, modelPath, vocabularyPath, word2VecOrgPath):
    begintime = time()
    sentences = GetSentence(trainDataPath)
    #对训练数据的参数配置
    model = gensim.models.Word2Vec(sentences,
                                   size = 100,
                                   window = 5,
                                   min_count = 1,
                                   workers = 4)
    loginformation("", "info", "The process is saving the model.")
    try:
        model.save(modelPath)
        model.wv.save_word2vec_format(word2VecOrgPath,
                                  vocabularyPath,
                                  binary=False)
    except:
        loginformation("", "info", "Error occurs while saving the word vector model!!")
    else:
        loginformation("", "info", "Finished the process of saving word vector model.")
    endtime = time()
    loginformation("", "info", "Total word vector training time is: " + (endtime - begintime))


#get the word vector of sentence after segment, requirement: space between word and word
def getSentenceVector(sentence, wordVectorModel, wordVectorModelDim):
    sentenceList = sentence.rstrip('\n').split("  ")
    sentenceWordVectorList = []
    for word in sentenceList:
        wordVector = []
        if(word != ' ' or word != ''):
            try:
                for i in range(wordVectorModelDim):
                    wordVector.append(str(wordVectorModel[word][i].strip))
            except KeyError:
                loginformation("", "debug", word + "not in the vocabulary !!!")
                for i in range(wordVectorModelDim):
                    wordVector.append("0.0")
                pass
            else:
                loginformation("", "info", "Get the vector of " + word)
        else:
            loginformation("", "info", "null word occrs")
        sentenceWordVectorList.append(wordVector)
    loginformation("", "info", "transfered sentence " + sentence + " to word vector.")
    return sentenceWordVectorList 


if __name__ == "__main__":
    rootPath = os.path.abspath(".") + "/"
    trainDataPath = rootPath + "../../../../../Documents/weibocluster_segmentByPynlpir/"
    modelPath = rootPath + "../../../data/preprocess/word2vectormodel/dim100Window5Min1.model"
    vocabularyPath = rootPath + "../../../data/preprocess/wordVectorVocabulary/dim100Window5Min1Vocabulary"
    word2VecOrgPath = rootPath + "../../../data/preprocess/word2VecOrgPath/dim100Window5Min1Word2VecOrgPath"
    wordVectorTrainModel(trainDataPath, modelPath, vocabularyPath, word2VecOrgPath)
