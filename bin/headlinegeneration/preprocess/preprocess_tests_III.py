__author__ = 'Chi'

import segment
import pynlpir
from bin.headlinegeneration.preprocess import sparseswords
import os

#
if __name__ == "__main__":
    #sentence = "郑州大学入选国家双一流大学"
    #print(segment.segment(sentence))    #test segment
    #print(segment.segment_tagging(sentence))    #test segment and tagging
    #对训练语料和测试语料进行分词
    """
    file_path = "../../../data/originaldata/PART_I.txt"
    segment_file_path = "../../../data/preprocess/PART_I_segment.txt"
    pynlpir.open()    
    with open(file_path, 'r', encoding='utf8') as file, open(segment_file_path, 'w') as seg_file:
        for line in file:
            try:
                seg_file.write(" ".join(str(i) for i in pynlpir.segment(line, pos_tagging=False)) + "\n")
            except UnicodeDecodeError:
                seg_file.write("occurring UnicodeDecodeError\n")
                pass
                #print(line + "  occur UnicodeDecodeError")
    pynlpir.close()
    """

    #
rootPath = os.path.abspath(".") + "/"
'''
#test the method of filterTrainingFile(training_file, after_filter_file)

training_file = rootPath + "../../../data/preprocess/PART_III_segment.txt"
after_filter_file = rootPath + "../../../data/preprocess/PART_III_segment_filter.txt"
sparseswords.filterTrainingFile(training_file, after_filter_file)
'''

'''
test the method of calculate() for count word frequency
data_file = rootPath + "/../../../data/preprocess/PART_III_segment_filter.txt"
statistical_result_file = rootPath + "/../../../data/preprocess/PART_III_segment_calculate.txt"
sparse_word_file = rootPath + "/../../../data/preprocess/PART_III_segment_sparsewordfile.txt"
sparseswords.calculate(data_file, statistical_result_file, sparse_word_file)
'''

'''
#test methods of getThreeClassSpareWord() and getUNK()
print(getThreeClassOfSpareWord("1232adfs搭刷歌哈哈哈"))
print(getUNK("1232adfs搭刷歌哈哈哈"))
'''

'''
#test the method of getSparesWordList()
sparseWordFile = rootPath + "/../../../data/preprocess/PART_I_segment_sparsewordfile.txt"
getSparseWordList(sparseWordFile)
'''

#"""
#test the method of replaceSparseWord()
sparseWordFile = rootPath + "/../../../data/preprocess/PART_III_segment_sparsewordfile.txt"
originalFile = rootPath + "/../../../data/preprocess/PART_III_segment_filter.txt"
afterReplaceFile = rootPath + "/../../../data/preprocess/PART_III_segment_replace.txt"
sparseswords.replaceSparseWord(originalFile, afterReplaceFile, sparseWordFile)
#"""

