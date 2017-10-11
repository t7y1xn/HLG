__author__ = 'Chi'

from time import time
from bin.headlinegeneration.utils.loginfor import loginformation
from gensim.models import word2vec
import re
import multiprocessing
import gensim
import os


def cleandata(rawdata):
    cleanr = re.compile('<.*>?>')
    cleantext = re.sub(cleanr, ' ', rawdata)
    return cleantext

#生成迭代器，以防止数据太大，内存不够
class GetSentence(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):        #__iter__方法的对象是可迭代的。
        for file in os.listdir(self.dirname):
            filePath = self.dirname + '/' + file
            for line in open(filePath):
                sline = line.strip()
                if sline == "":
                    continue
                rline = cleandata(sline)
                yield rline.split(" ")

def wordVectorTrainModel(trainDataPath, modelPath, vocabularyPath, word2VecOrgPath):
    begintime = time()
    sentences = GetSentence(trainDataPath)
    #对训练数据的参数配置
    model = gensim.models.Word2Vec(sentences,
                                   size = 200,
                                   window = 10,
                                   min_count = 1,
                                   workers = 4)
    model.save(modelPath)
    model.wv.save_word2vec_format(word2VecOrgPath,
                                  vocabularyPath,
                                  binary=False)
    endtime = time()
    loginformation("", "info", "Total training time is: " + (endtime - begintime))
