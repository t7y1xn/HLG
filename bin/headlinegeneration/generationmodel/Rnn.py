import logging
import numpy as np
import os
import pickle
import sys
from bin.headlinegeneration.utils.loginfor import loginformation
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import gensim

#set dimension
INPUTDIMENSIONS = [1, 250]  # dimension of input features
EIHW = [250, 50]   # weight matrix between input and hidden layer of encoder
EIHB = [1, 50]     # bias matrix between input and hidden layer of encoder
EHOW = [50, 50]     # weight matrix between hidden and input layer of encoder
EHOB = [1, 50]     # bias matrix between hidden and output layer of encoder
DIHW = [250, 50]   # weight matrix between input and hidden layer of decoder
DIHB = [1, 50]   # bias matrix between input and hidden layer of decoder
DHOW = [50, 200]   # weight matrix between hidden and output layer of decoder
DHOB = [1, 200]   # bias matrix between hidden and output layer of decoder

# Get the vector list of sentence;
def GetWordVector(sentence_list, wordVectorModel):
    UNK = []
    for i in range(200):
        UNK.append(0.5)
    sentenceWordVectorList = []
    for word in sentence_list:
        if(word != ' ' or word != ''):
            try:
                sentenceWordVectorList.append(wordVectorModel[word])
            except KeyError:
                print(word + " is not in vocabulary!!")
                sentenceWordVectorList.append(UNK)
    return sentenceWordVectorList

def SetWeight(shape, Name):       #set weight randomly
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=1.0 ),name=Name)

def SetBias(shape, Name):     #set the bias randomly
    return tf.Variable(tf.random_uniform(shape=shape, minval=-30, maxval=30, dtype=tf.float32), name=Name)

def EncoderInitial():   #initialize the encoder
    encoderWeights={
        'w1' : SetWeight(EIHW, 'w1'),
        'w2' : SetWeight(EHOW, 'w2')
    }
    encoderBias={
        'b1' : SetBias(EIHB, "b1"),
        'b2' : SetBias(EHOB, "b2")
    }
    return encoderWeights, encoderBias

def RnnEncoder(x, encoderWeights, encoderBias):       #encoder, from word to feature; forward
    y_1 = tf.nn.softmax(tf.add(tf.matmul(x, encoderWeights['w1']), encoderBias['b1']))
    y_2 = tf.nn.softmax(tf.add(tf.matmul(y_1, encoderWeights['w2']) , encoderBias['b2']))
    return  y_2

def DecoderInitial():   #Initialize the decoder
    decoderWeights = {
        'w1' : SetWeight(DIHW, "w1_decode"),
        'w2' : SetWeight(DHOW, "w2_decode")
    }
    decoderBias = {
        'b1' : SetBias(DIHB, "b1_decode"),
        'b2' : SetBias(DHOB, "b2_decode")
    }
    return decoderWeights, decoderBias


def RNNdecoder(y_2, decoderWeights, decoderBias):    #decode, output vector, and then get word
    y_1_decode = tf.nn.softmax(tf.add(tf.matmul(y_2, decoderWeights['w1']), decoderBias['b1']))
    y_2_decode = tf.nn.softmax(tf.add(tf.matmul(y_1_decode, decoderWeights['w2']), decoderBias['b2']))
    return y_2_decode

def GetPickl(picklFile):
    AllList = []
    with open(picklFile, 'rb') as pf:
        AllList = pickle.load(pf)
    return AllList

def PredictWord(decoderOut, textVectorList):
    index = 0
    Out = decoderOut.eval()[0]
    MIN = sys.maxsize
    for num in range(len(textVectorList)):
        wordVector = textVectorList[num]
        diff = 0.0
        for i in range(len(wordVector)):
            diff += abs(wordVector[i] - Out[i])
        if MIN > diff:
            MIN = diff
            index = num
    return tf.expand_dims(tf.convert_to_tensor(textVectorList[index]), 0), index

def TrainingModel(dataList, encoderWeights, encoderBias, decoderWeights, decoderBias, wordVectorModel, learningRate):
    encoderInput = tf.placeholder(tf.float32, shape=[1, 200])
    encoderOut = tf.Variable(tf.zeros(EHOB))
    #dX = tf.concat([encoderInput, encoderOut], axis=1)
    #encoderOut = RnnEncoder(dX, encoderWeights, encoderBias)
    decoderOut = tf.Variable(tf.zeros(DHOB))
    predictWord = tf.Variable(tf.ones(DHOB))
    #acculateLoss = tf.Variable(tf.zeros([]))
    #trainOp = tf.train.AdamOptimizer(learningRate).minimize(acculateLoss)
    tf.variables_initializer([decoderOut, predictWord, encoderOut])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for item in dataList:
            textList = item['text:'].split(" ")
            titleList = item['title:'].split(" ")
            textVectorList = GetWordVector(textList, wordVectorModel)
            titleVectorList = GetWordVector(titleList, wordVectorModel)
            #encoderOut = tf.Variable(tf.zeros(EHOB))
            for wordItem in textVectorList:     #encoder
                encoderInput = tf.expand_dims(tf.convert_to_tensor(wordItem), 0)
                #X = tf.concat([encoderInput, encoderOut], axis=1)
                encoderOut = RnnEncoder(tf.concat([encoderInput, encoderOut], axis=1), encoderWeights, encoderBias)

            #decoderOut = tf.Variable(tf.zeros(DHOB))
            #predictWord = tf.Variable(tf.ones(DHOB))
            dX = tf.concat([encoderOut, tf.div(tf.add(decoderOut, predictWord), 2)], axis=1)

            genTitleList = []
            genTitleVector = []
            #acculateLoss = tf.Variable(tf.zeros([]))
            #trainOp = tf.train.AdamOptimizer(learningRate).minimize(acculateLoss)

            for i in range(len(titleVectorList)):   #解码器
                dX = tf.concat([encoderOut, tf.div(tf.add(decoderOut, predictWord), 2)], axis=1)
                decoderOut = RNNdecoder(dX, decoderWeights, decoderBias)
                trainOp = tf.train.AdamOptimizer(learningRate).minimize(acculateLoss)
                #sess.run(tf.global_variables_initializer())
                predictWord, index = PredictWord(decoderOut, textVectorList)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=titleVectorList[i], logits=predictWord)
                L = tf.reshape(loss, [])
                genTitleList.append(textList[index])
                acculateLoss = tf.add(acculateLoss, tf.reshape(loss, []))
            sess.run(trainOp)
            print(genTitleList, end='')
            print(titleList)


if __name__ == "__main__":
    #'''
    rootPath = os.path.abspath("./") + "/"
    pickleFile = rootPath + "../../../data/preprocess/pickledata/PART_III_segment.pkl"
    dataList = GetPickl(pickleFile)
    wordVectorModel = gensim.models.Word2Vec.load(
        rootPath + "../../../../NeuralHeadlineGeneration/data/dataWordVector/PARTDATA_dim200_window10_mincount1.model")
    encoderWeights, encoderBias = EncoderInitial()
    decoderWeights, decoderBias = DecoderInitial()
    TrainingModel(dataList, encoderWeights, encoderBias, decoderWeights, decoderBias, wordVectorModel, learningRate=0.01)
    #'''
    '''
    learning_rate = 0.01
    #np数组构造连接，然后转为tensor
    #prefixones = np.ones([1,50])
    #prefixzeros = np.zeros([1, 50])
    #prefixones = np.ndarray.astype(prefixones,dtype='float32')
    #prefixzeros = np.ndarray.astype(prefixzeros, dtype='float32')
    
    #In = np.hstack((prefixzeros, prefixones))
    #IN = tf.convert_to_tensor(In, dtype=tf.float32)
    #
    encoderWeights, encoderBias = EncoderInitial()
    decoderWeights, decoderBias = DecoderInitial()
    y_ = tf.placeholder(tf.float32, shape=[1, 200])  # place holder, feature dimension of output word
    x_2_decode = tf.placeholder(tf.float32, shape=[1, 250])
    y_2_decode = tf.placeholder(tf.float32, shape=[1, 200])
    y_2 = tf.placeholder(tf.float32, shape=[1, 50])
    first_input = np.zeros((1, 50))  # genenrate the first word input, to complement dimension(补充维数)。
    prefixzerosDecode = tf.Variable(tf.zeros([1, 200]))
    #
    tf.summary.histogram("Weight1ofEncode", encoderWeights['w1'])
    tf.summary.histogram("Bias1ofEncode", encoderBias['b1'])
    tf.summary.histogram("Weight2ofEncode", encoderWeights['w2'])
    tf.summary.histogram("Bias2ofEncode", encoderBias['b2'])

    #
    x = tf.Variable(tf.random_uniform([1, 200], minval=-0.1, maxval=0.1, dtype=tf.float32), name="inputFeature")
    prefixones = tf.Variable(tf.ones([1, 50]), name="prefixones")
    prefixzeros = tf.Variable(tf.zeros([1, 50]), name="prefixzeros")
    X = tf.concat([x, prefixzeros], axis=1)  # 将两个tensor进行拼接
    #
    textList = []
    for i in range(100):
        textList.append(tf.Variable(tf.random_uniform([1, 200], minval=-0.1, maxval=0.1, dtype=tf.float32)))
    titleList = []
    for i in range(10):
        titleList.append(tf.Variable(tf.random_uniform([1, 200], minval=-0.1, maxval=0.1, dtype=tf.float32)))
    #
    #RnnEncoder(X, w1, w2, b1, b2)
    with tf.Session() as sess:
        y_1, y_2 = RnnEncoder(X, encoderWeights, encoderBias)
        tf.summary.histogram("y_2", y_2)
        for i in range(1, len(textList)):  #编码器
            X = tf.concat([x, y_2], axis=1)
            y_1, y_2 = RnnEncoder(X, encoderWeights, encoderBias)


        In = tf.concat([y_2, prefixzerosDecode], axis=1)
        y_2_decode, y_1_decode = RNNdecoder(In, decoderWeights, decoderBias)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=titleList[0], logits=y_2_decode)
        L = tf.reshape(loss, [])
        acculateLoss = L
        for i in range(1, len(titleList)):  #解码器
            In = tf.concat([y_2, y_2_decode], axis=1)
            y_2_decode, y_1_decode = RNNdecoder(In, decoderWeights, decoderBias)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=titleList[i], logits=y_2_decode)
            L = tf.reshape(loss, [])

            acculateLoss = tf.add(acculateLoss, L)
            trainOp = tf.train.AdamOptimizer(learning_rate).minimize(acculateLoss)
        tf.summary.scalar('acculate_loss', acculateLoss)
        tf.summary.histogram("y_2_decode", y_2_decode)
        merged = tf.summary.merge_all()
        test = tf.summary.FileWriter("./testlog/", sess.graph)
        sess.run(tf.global_variables_initializer())
        for num in range(200):
            if (num % 2 == 0):
                sumDecode = sess.run(merged)
                print(sess.run(L))
                test.add_summary(sumDecode, i)
            trainOp.run()

            #sess.run(tf.local_variables_initializer())

        #sess.run(trainOp)
        #print(sess.run(y_2_decode))
    test.close()
    print()
    #'''