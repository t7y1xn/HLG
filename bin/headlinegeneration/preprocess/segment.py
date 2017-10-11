__author__ = 'Chi'

import pynlpir


#tokenize
def segment(sentence):
    pynlpir.open()  #Initializes the NLPIR API
    sentence_segment = pynlpir.segment(sentence, pos_tagging= False)    #
    pynlpir.close() #Exits the NLPIR and frees allocated memory.
    return sentence_segment

#tokenize and tagging
def segment_tagging(sentence):
    pynlpir.open()  # Initializes the NLPIR API
    sentence_segment_tag = pynlpir.segment(sentence)    #Get the result of segment and tagging
    pynlpir.close()  # Exits the NLPIR and frees allocated memory.
    return sentence_segment_tag         #return results