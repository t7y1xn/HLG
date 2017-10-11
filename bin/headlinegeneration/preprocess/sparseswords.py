__author__ = 'Chi'

import collections


#calculate words number of file
def calculate(data_file, statistical_result_file, sparse_word_file):
    with open(data_file, 'r') as df, open(statistical_result_file, 'w') as sf, open(sparse_word_file, 'w') as swf:
        word_box = []
        for line in df:
            word_box.extend(line.rstrip('\n').split(" "))
        for i in collections.Counter(word_box).items():
            print(i)
            (a, b) = i
            if(b > 3):
                sf.write(a+":"+str(b)+"\n")
            else:
                swf.write(a+":"+str(b)+"\n")
        print("finished!!!")

#the method is used to get sparse word list
def getSparseWordList(spareseWordFile):
    sparse_word_list = []
    with open(spareseWordFile, 'r') as swf:
        for line in swf:
            num = 0
            for i in line.split(":")[0]:
                if (i >= u'\u0041' and i <= u'\u005a') or (i >= u'\u0061' and i <= u'\u007a') \
                        or (i >= u'\u0030' and i <= u'\u0039'):
                    num += 1
            if(num > len(line.split(":")[0])/2):
                continue
            else:
                sparse_word_list.append(line.split(":")[0])
                #print(line.split(":")[0])
    #print(len(sparse_word_list))
    return sparse_word_list

#filter the training file after segment
def filter_training_file(training_file, after_filter_file):
    rownum = 0
    title = ""
    text = ""
    with open(training_file, 'r') as df, open(after_filter_file, 'w') as sf:
        for line in df:
            rownum += 1
            if (rownum % 3 == 1):
                continue
            elif (rownum % 3 == 2):
                sf.write(line.lstrip("abstract : "))
            else:
                sf.write(line.lstrip("text : "))

"""
The method getThreeClassSpareWord() is used to get three class label of spare words: Digits class, Characters class and others
"""
def getThreeClassOfSpareWord(word):
    digitNumber = 0
    characterNumber = 0
    othersNumber = 0
    for index in range(len(word)):
        #Judge whether character is digit? and it can also use the condition: (word[index] >= u'\u0030' and word[index] <=u'\u0039') to judge
        if(word[index].isdigit()):
            digitNumber += 1
            continue
        if((word[index] >= u'\u0041' and word[index] <= u'\u005a') \
                or (word[index] >= u'\u0061' and word[index] <= u'\u007a')):
            characterNumber += 1
            continue
        othersNumber += 1
    classLabel = "Others"
    if digitNumber > characterNumber and digitNumber > othersNumber:
        classLabel = "Digits"
    if characterNumber > digitNumber and digitNumber > othersNumber:
        classLabel = "Characters"
    return classLabel

'''
The method getUNK() is used to judge UNK word
'''
def getUNK(word):
    digitOrCharacterNumber = 0
    UNK = False
    for index in range(len(word)):
        # Judge english characters or digits
        if (word[index] >= u'\u0041' and word[index] <= u'\u005a') \
                or (word[index] >= u'\u0061' and word[index] <= u'\u007a') \
                or (word[index] >= u'\u0030' and word[index] <=u'\u0039'):
            digitOrCharacterNumber += 1
    #print(digitOrCharacterNumber)
    if digitOrCharacterNumber > len(word)/2:
        UNK = True
    return UNK

'''
The method of replaceSparesWord() is used to replace the spares word in original file,
originalFile is the file of original file after segment
afterReplaceFile is the file of replace sparse words in original file
spareWordFile is the file of storing sparse words
'''
def replaceSparseWord(originalFile, afterReplaceFile, sparseWordFile):
    sparseWordList = getSparseWordList(sparseWordFile)
    with open(originalFile, 'r') as of, open(afterReplaceFile, 'w') as arf:
        for line in of:
            for word in line.split(" "):
                if(word in sparseWordList):
                    arf.write("UNK" + " ")
                elif(getUNK(word)):
                    arf.write("UNK" + " ")
                else:
                    arf.write(word + " ")

'''
#test the method of calculate() for count word frequency
data_file = "../../../data/preprocess/PART_I_segment_temp.txt"
statistical_result_file = "../../../data/preprocess/PART_I_segment_temp_calculate.txt"
sparse_word_file = "../../../data/preprocess/SPARSE_WORD_FILE.txt"
calculate(data_file, statistical_result_file, sparse_word_file)
'''

'''
#test methods of getThreeClassSpareWord() and getUNK()
print(getThreeClassOfSpareWord("1232adfs搭刷歌哈哈哈"))
print(getUNK("1232adfs搭刷歌哈哈哈"))
'''

'''
#test the method of getSparesWordList()
sparseWordFile = "../../../data/preprocess/SPARSE_WORD_FILE.txt"
getSparseWordList(sparseWordFile)
'''

#test the method of replaceSparseWord()
sparseWordFile = "../../../data/preprocess/SPARSE_WORD_FILE.txt"
originalFile = "../../../data/preprocess/PART_I_segment_temp_filter.txt"
afterReplaceFile = "../../../data/preprocess/PART_I_segment_temp_replace.txt"
replaceSparseWord(originalFile, afterReplaceFile, sparseWordFile)
