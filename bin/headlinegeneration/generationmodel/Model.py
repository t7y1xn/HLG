__author__ = 'Chi'

import tensorflow as tf
import sys
import pickle
import gensim
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'      #

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

def SetWeight(shape, Name):       #set weight randomly
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=1.0 ),name=Name)

def SetBias(shape, Name):     #set the bias randomly
    return tf.Variable(tf.random_uniform(shape=shape, minval=-30, maxval=30, dtype=tf.float32), name=Name)

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

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def encoder():
  encoderVariable = {'weights1': SetWeight(EIHW, 'weights1'),
                     'weights2': SetWeight(EHOW, 'weights1'),
                     'biases1': SetBias(EIHB, "biases1"),
                     'biases2': SetBias(EIHB, "biases2")
                     }
  '''
    with tf.name_scope("encoder"):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights1'):
      encoderVariable['weights1'] = tf.Variable(tf.random_normal([EIHW], stddev=1.0), name='weights1')
      variable_summaries(encoderVariable['weights1'])
    with tf.name_scope('bias1'):
      encoderVariable['biases1'] = tf.Variable(tf.random_uniform([EIHB], minval=-1.0, maxval=1.0))
      variable_summaries(encoderVariable['biases1'])
    with tf.name_scope('weights2'):
      encoderVariable['weights2'] = tf.Variable(tf.random_normal([EHOW], stddev=1.0))
      variable_summaries(encoderVariable['weights2'])
    with tf.name_scope('bias2'):
      encoderVariable['biases2'] = tf.Variable(tf.random_uniform([EHOB], minval=-1.0, maxval=1.0))
      variable_summaries(encoderVariable['biases2'])
    with tf.name_scope('encoderOut'):
      encoderVariable['encoderOut'] = tf.Variable(tf.zeros([EHOB]))
  '''
  return encoderVariable

def decoder():
    decoderVariable = {'weights1': SetWeight(DIHW, 'weights1'),
                       'weights2': SetWeight(DHOW, 'weights1'),
                       'biases1': SetBias(DIHB, "biases1"),
                       'biases2': SetBias(DHOB, "biases2"),
                       'decoderOut': tf.Variable(tf.zeros(DHOB))
                      }
    '''
    decoderVariable = {}
    with tf.name_scope("decoder"):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights1'):
            decoderVariable['weights1'] = tf.Variable(tf.random_normal([DIHW], stddev=1.0))
            variable_summaries(decoderVariable['weights1'])
        with tf.name_scope('bias1'):
            decoderVariable['biases1'] = tf.Variable(tf.random_uniform([DIHB], minval=-1.0, maxval=1.0))
            variable_summaries(decoderVariable['biases1'])
        with tf.name_scope('weights2'):
            decoderVariable['weights2'] = tf.Variable(tf.random_normal([DHOW], stddev=1.0))
            variable_summaries(decoderVariable['weights2'])
        with tf.name_scope('bias2'):
            decoderVariable['biases2'] = tf.Variable(tf.random_uniform([DHOB], minval=-1.0, maxval=1.0))
            variable_summaries(decoderVariable['biases2'])
        with tf.name_scope('decoderOut'):
            decoderVariable['decoderOut'] = tf.Variable(tf.zeros([DHOB]))
    '''
    return decoderVariable

def encoderCell(X, encoderVariable):
    y_1 = tf.nn.softmax(tf.add(tf.matmul(X, encoderVariable['weights1']), encoderVariable['biases1']))
    y_2 = tf.nn.softmax(tf.add(tf.matmul(y_1, encoderVariable['weights2']), encoderVariable['biases1']))
    return y_2

def decoderCell(dX, decoderVariable):
    y_1_decode = tf.nn.softmax(tf.add(tf.matmul(dX, decoderVariable['weights1']), decoderVariable['biases1']))
    y_2_decode = tf.nn.softmax(tf.add(tf.matmul(y_1_decode, decoderVariable['weights2']), decoderVariable['biases2']))
    return y_2_decode

def runEncoder(textVectorList, encoderVariable, wordVectorModel, encoderOut):
    for i in range(len(textVectorList)):
        encoderOut = encoderCell(tf.concat([tf.expand_dims(tf.convert_to_tensor(textVectorList[i]), 0),
                                                              encoderOut], axis=1), encoderVariable)
    return encoderOut

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


def GetPickl(picklFile):
    AllList = []
    with open(picklFile, 'rb') as pf:
        AllList = pickle.load(pf)
    return AllList


def runModel(learningRate):
    encoderVariable = encoder()
    decoderVariable = decoder()
    Y = tf.placeholder(tf.float32, DHOB)
    predictWord = tf.Variable(tf.ones(DHOB))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predictWord)
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(loss)
    with tf.name_scope('train'):
        trainStep = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)
    encoderOut = tf.Variable(tf.zeros(EHOB))
    decoderOut = tf.Variable(tf.zeros(DHOB))
    rootPath = os.path.abspath("./") + "/"
    pickleFile = rootPath + "../../../data/preprocess/pickledata/PART_III_segment.pkl"
    dataList = GetPickl(pickleFile)
    wordVectorModel = gensim.models.Word2Vec.load(
        rootPath + "../../../../NeuralHeadlineGeneration/data/dataWordVector/PARTDATA_dim200_window10_mincount1.model")

    merged = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter('log')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for item in dataList:
            titleList = item['title:'].split(" ")
            titleVectorList = GetWordVector(titleList, wordVectorModel)
            textList = item['text:'].split(" ")
            textVectorList = GetWordVector(textList, wordVectorModel)
            encoderOut = runEncoder(textVectorList=textVectorList, encoderVariable=encoderVariable,
                                    wordVectorModel=wordVectorModel, encoderOut=encoderOut)
            generateTitleList = []
            for title in titleVectorList:
                decoderCell(tf.concat([encoderOut, tf.div(tf.add(decoderVariable['decoderOut'], predictWord), 2)], axis=1), decoderVariable)
                predictWord, index = PredictWord(decoderVariable['decoderOut'], textVectorList=textVectorList)
                generateTitleList.append(textList[index])
                merge, _ = sess.run([merged, trainStep], feed_dict={Y:np.array(title).reshape(DHOB)})
                trainWriter.add_summary(merge)
            exit()
    trainWriter.close()
if __name__ == "__main__":
    learningRate = 0.01
    runModel(learningRate=learningRate)
    #Y = tf.placeholder(tf.float32, DHOB)
    #title = np.random.normal(size=[200])
    #with tf.Session() as sess:
    #    print(sess.run(Y, feed_dict={Y:title.reshape([1,200])}))