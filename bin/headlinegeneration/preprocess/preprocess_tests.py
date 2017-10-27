__author__ = 'Chi'

from bin.headlinegeneration.preprocess.segment import segment
import pynlpir
import os

#
if __name__ == "__main__":
    #sentence = "郑州大学入选国家双一流大学"
    #print(segment.segment(sentence))    #test segment
    #print(segment.segment_tagging(sentence))    #test segment and tagging
    #对训练语料和测试语料进行分词
    rootPath = os.path.abspath(".")
    file_path = rootPath + "/../../../data/originaldata/PART_III.txt"
    segment_file_path = rootPath + "/../../../data/preprocess/PART_III_segment.txt"
    pynlpir.open() 
    print(rootPath)
    with open(file_path, 'r', encoding='utf8') as file, open(segment_file_path, 'w') as seg_file:
        for line in file:
            try:
                seg_file.write(" ".join(str(i) for i in pynlpir.segment(line, pos_tagging=False)) + "\n")
            except UnicodeDecodeError:
                seg_file.write("occurring UnicodeDecodeError\n")
                pass
    pynlpir.close()
