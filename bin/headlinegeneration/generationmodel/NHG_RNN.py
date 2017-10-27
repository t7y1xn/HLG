__author__ = 'Chi'

import logging
import numpy as np
import os
from bin.headlinegeneration.utils.loginfor import loginformation
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import gensim


# Get the vector list of sentence;
def GetWordVector(sentence_list, wordVectorModel):
    sentenceWordVectorList = []
    for word in sentence_list:
        sentenceWordVectorList.append(wordVectorModel[word])
    return sentenceWordVectorList

def SetWeight(shape, Name):       #set weight randomly
    return tf.Variable(tf.random_normal(shape=shape, stddev=1),name=Name)

def SetBias(shape, Name):     #set the bias randomly
    return tf.Variable(tf.constant(shape=shape, value=0.1), name=Name)

def EncoderInitial():   #initialize the encoder 
    w1 = SetWeight([250, 50],"w1")   #get the weight matirx of input and hidden layer randomly
    w2 = SetWeight([50, 50], "w2")    #get the weight matirx of hidden and output layer
    b1 = SetBias([1, 50],"b1")       #get bias of input and hidden layer randomly
    b2 = SetBias([1, 50],"b2")       #get bias of hidden and output layer randomly
    return w1, w2, b1, b2

def RnnEncoder(x, w1, w2, b1, b2):       #encoder, from word to feature; forward
    y_1 = tf.placeholder(tf.float32, [1,50])
    y_1 = tf.nn.softmax(tf.matmul(x, w1) + b1)
    y_2 = tf.nn.softmax(tf.matmul(y_1, w2) + b2)
    return y_2

def DecoderInitial():   #Initialize the decoder
    w1_decode = SetWeight([250, 50],"w1_decode")    # get the weight matirx of first and hidden 
    w2_decode = SetWeight([50, 200],"w2_decode")    # get the weight matirx of hidden and last
    b1_decode = SetBias([1, 50],"b1_decode")    #get the bias matirx of first and hidden
    b2_decode = SetBias([1, 200],"b2_decode")   # get the bias matirx of hidden and last
    return w1_decode, w2_decode, b1_decode, b2_decode

def RNNdecoder(y_2, w1_decode, w2_decode, b1_decode, b2_decode):    #decode, output vector, and then get word
    y_1_decode = tf.placeholder(tf.float32, [1,50])
    y_1_decode = tf.nn.softmax(tf.matmul(y_2, w1_decode) + b1_decode)
    y_2_decode = tf.nn.softmax(tf.matmul(y_1_decode, w2_decode) + b2_decode)
    return y_2_decode


def RunModel(dataPath,wordVectorModel):
    w1, w2, b1, b2 = EncoderInitial()                               #initialize the weight and bias of encoder
    w1_decode, w2_decode, b1_decode, b2_decode = DecoderInitial()   #initialize the weight and bias of decoder
    x = tf.placeholder(tf.float32, shape = [1, 250])        #place holder, feature dimension of input word
    y_ = tf.placeholder(tf.float32, shape=[1, 200])         # place holder, feature dimension of output word
    x_2_decode = tf.placeholder(tf.float32, shape=[1,250])
    y_2_decode = tf.placeholder(tf.float32,shape=[1,200])
    y_2 = tf.placeholder(tf.float32, shape=[1,50])
    first_input = np.zeros((1,50))   #genenrate the first word input, to complement dimension(补充维数)。

    #accumulate loss
    #accumulate_loss = tf.placeholder(tf.float32, shape=[1,0])
    accumulate_loss = 0
    tf.summary.scalar("loss", accumulate_loss)
    #read the data of training file
    number = 1
    computation_graph = tf.InteractiveSession()
    sess = tf.Session()
    merged = tf.summary.merge_all()
    computation_graph.run()
    summary_writer = tf.summary.FileWriter("./log/")
    title_word_vector_list = []
    for line in open(dataPath):
        loginformation("", "info", "the line number is: " + str(number))
        if number % 3 == 1:
            number += 1
            continue
        if number % 3 == 2: #get title vector
            line_list = line.rstrip("\n").split(" ")
            title_word_vector_list = GetWordVector(line_list, wordVectorModel)
            number += 1
            continue
        print(accumulate_loss)
        if number % 10 == 0:
            print(accumulate_loss)
            summary, loss = sess.run([merged, accumulate_loss], feed_dict=feed_dict(False))
            summary_writer.add_summary("./log/", number)
            loginformation("", "info", "loss is: " + str(accumulate_loss))
        if number % 3 == 0:
            line_list = line.rstrip("\n").split(" ")
            wordVectorList = GetWordVector(line_list,wordVectorModel)
            y_2_value = np.zeros((1,50))
            y_2_value = np.ndarray.astype(y_2_value,dtype='float32')
            for wordIndex in range(len(wordVectorList)):
                if wordIndex == 0:
                    X = np.hstack((wordVectorList[wordIndex], y_2_value[0]))
                    X = X.reshape([1,250])
                    X = np.ndarray.astype(X, dtype='float32')
                    y_2 = RnnEncoder(x.eval(feed_dict={x:X}), w1, w2, b1, b2)
                    #with sess.as_default():
                        #sess.run(y_2)
                else:#revised here
                    #with sess.as_default():
                    X = np.hstack((wordVectorList[wordIndex], y_2[0].eval()))
                    X = X.reshape([1,250])
                    X = np.ndarray.astype(X, dtype='float32')
                    y_2 = RnnEncoder(x.eval(feed_dict={x:X}), w1, w2, b1, b2)
                    #sess.run(y_2)
            y_2_decode_ = np.zeros([1,200])
            y_2_decode_ = np.ndarray.astype(y_2_decode_,dtype='float32')
            for titleWordIndex in range(len(title_word_vector_list)):
                if titleWordIndex == 0:
                    X_2 = np.hstack((y_2[0].eval(), y_2_decode_[0]))
                    X_2 = X_2.reshape([1,250])
                    X_2 = np.ndarray.astype(X_2,dtype='float32')
                    #with sess.as_default():
                    y_2_decode = RnnEncoder(x_2_decode.eval(feed_dict={x_2_decode:X_2}), w1_decode, w2_decode, b1_decode, b2_decode)
                    #sess.run(y_2_decode)

                else:
                    X_2 = np.hstack((y_2[0].eval(), y_2_decode[0].eval()))
                    X_2 = X_2.reshape([1,250])
                    X_2 = np.ndarray.astype(X_2,dtype='float32')
                    y_2_decode = RnnEncoder(x_2_decode.eval(feed_dict={x_2_decode:X_2}), w1_decode, w2_decode, b1_decode, b2_decode)
                    #sess.run(y_2_decode)
                #计算累积损失
                titleword = title_word_vector_list[titleWordIndex].reshape([1,200])
                accumulate_loss = accumulate_loss + tf.reduce_mean(tf.square(titleword-y_2_decode))
            trainstep = tf.train.AdamOptimizer(1e-4).minimize(accumulate_loss)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(trainstep)
            #trainstep.run(feed_dict={x:X, x_2_decode:X_2, y_:title_word_vector_list[titleWordIndex]})
            number += 1
    sess.close()
    summary_writer.close()
    computation_graph.close()
'''
class RnnModel():
    def __init__(self, batch_size, x_dimension, y_dimension, hiddensize):
        self.batch_size = batch_size        #批处理大小
        self.x_dimension = x_dimension      #输入的维度
        self.y_dimension = y_dimension      #输出的维度
        self.hiddensize = hiddensize        #隐含层的节点大小
        self.input_x = tf.placeholder(tf.float32, [batch_size, x_dimension])    #占位符，输入数据
        self.target_y = tf.placeholder(tf.float32, [batch_size, y_dimension])   #占位符，输出数据
        self.state_size = tf.placeholder(tf.float32, [batch_size, hiddensize])
        #rnn_cell = tf.contrib.rnn.BasicRNNCell(self.hiddensize)
        #rnn_cell.__init__(num_units=50, input_size=None, activation=tf.softmax)
'''

def StoreAllVariable(_w1, _w2, _b1, _b2, _w1_decode, _w2_decode, _b1_decode, _b2_decode, parameterfile):
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, shape = [1, 250])  
        y_ = tf.placeholder(tf.float32, shape=[1, 200])
        w1 = tf.constant(_w1, name='w1')
        w2 = tf.constant(_w2, name='w2')
        b1 = tf.constant(_b1, name='w2')
        b2 = tf.constant(_b2, name='w2')


        x_2_decode = tf.placeholder(tf.float32, shape=[1,250])
        y_2_decode = tf.placeholder(tf.float32,shape=[1,200])
        w1_decode = tf.constant(_w1_decode, name='w2')
        w2_decode = tf.constant(_w2_decode, name='w2')
        b1_decode = tf.constant(_b1_decode, name='w2')
        b2 = tf.constant(_b2_decode, name='w2')

        sess = tf.Session()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)


        graph_def = g.as_graph_def()
        tf.train.write_graph(graph_def, parameterfile)
        sess.close()
    return



if __name__ == "__main__":
    rootPath = os.path.abspath(".") + "/"
    loginformation("", "info","Training Rnn Model")
    loginformation("", "info", "Get word vector infomation")
    dataPath = rootPath + "../../../../NeuralHeadlineGeneration/data/originalData/PART_I_trainW2V_segment.txt"
    wordVectorModel = gensim.models.Word2Vec.load(rootPath + "../../../../NeuralHeadlineGeneration/data/dataWordVector/PARTDATA_dim200_window10_mincount1.model")
    RunModel(dataPath, wordVectorModel)

    '''
    y_1 = tf.placeholder(tf.float32, [1,50])
    y_s = np.zeros([1,50])
    y_s = np.ndarray.astype(y_s, dtype='float32')
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    with sess.as_default():
        print(y_1.eval(feed_dict={y_1:y_s}))
    sess.close()
    '''
