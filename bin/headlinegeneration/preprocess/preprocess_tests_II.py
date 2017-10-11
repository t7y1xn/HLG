__author__ = 'Chi'

#from bin.headlinegeneration.preprocess import segment
#from bin.headlinegeneration.utils import loginfor
import segment
import pynlpir

#
if __name__ == "__main__":
    #sentence = "郑州大学入选国家双一流大学"
    #print(segment.segment(sentence))    #test segment
    #print(segment.segment_tagging(sentence))    #test segment and tagging
    #对训练语料和测试语料进行分词
    file_path = "../../../data/originaldata/PART_I.txt"
    segment_file_path = "../../../data/preprocess/PART_I_segmentTemp.txt"
    pynlpir.open()    
    with open(file_path, 'r', encoding='utf8') as file, open(segment_file_path, 'w') as seg_file:
        for line in file:
            seg_file.write(" ".join(str(i) for i in pynlpir.segment(line, pos_tagging=False)) + "\n")
    pynlpir.close()