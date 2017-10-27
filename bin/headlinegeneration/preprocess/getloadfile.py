__author__ = 'Chi'

import pickle
from bin.headlinegeneration.utils.loginfor import loginformation

def storeDataFile(dataFile, pickleFile):
    AllList = []
    with open(dataFile, 'r') as df:
        num = 0
        id = 0
        preid = -1
        title = ""
        text = ""
        for line in df:
            num += 1
            try:
                if num % 3 == 1:
                    id = int(line.split(" ")[1])
                    #loginformation("", "info", line.split(" ")[1])
                elif num % 3  == 2:
                    title = line.lstrip("abstract : ").rstrip('\n')
                else:
                    text = line.lstrip("text : ").rstrip('\n')
                    if preid + 1 == id and title.strip() != "":
                        AllList.append({'text:': text, 'title:': title, 'id: ': id})
                        #loginformation("", "info", {'text:': text, 'title:': title, 'id: ': id})
                    else:
                        print("id" + str(id) + " title: " + title + " text: " + text)
                    preid = id

            except:
                loginformation("", "info", "Error occurs!!" + "text: " + text + " title: " + title + " id: " + id)
                pass
    print(len(AllList))
    with open(pickleFile, 'wb') as pf:
        pickle.dump(AllList, pf, -1)
    return

def storeUNKDataFile(dataFile, pickleFile):
    AllList = []
    with open(dataFile, 'r') as df:
        num = 0
        id = 0
        preid = -1
        title = ""
        text = ""
        for line in df:
            num += 1
            try:
                if num % 3 == 1:
                    id = int(line.split(" ")[1])
                elif num % 3 == 2:
                    UNKtitle = []
                    title = line.lstrip("abstract : ").rstrip('\n').split(" ")
                    for word in title:
                        digitOrCharacterNumber = 0
                        for index in range(len(word)):
                            # Judge english characters or digits
                            if (word[index] >= u'\u0041' and word[index] <= u'\u005a') \
                                    or (word[index] >= u'\u0061' and word[index] <= u'\u007a') \
                                    or (word[index] >= u'\u0030' and word[index] <= u'\u0039'):
                                digitOrCharacterNumber += 1
                        if digitOrCharacterNumber > len(word) / 2:
                            UNKtitle.append("UNk")
                        else:
                            UNKtitle.append(word)
                else:
                    text = line.lstrip("text : ").rstrip('\n').split(" ")
                    UNKtext = []
                    for word in text:
                        digitOrCharacterNumber = 0
                        for index in range(len(word)):
                            # Judge english characters or digits
                            if (word[index] >= u'\u0041' and word[index] <= u'\u005a') \
                                    or (word[index] >= u'\u0061' and word[index] <= u'\u007a') \
                                    or (word[index] >= u'\u0030' and word[index] <= u'\u0039'):
                                digitOrCharacterNumber += 1
                        if digitOrCharacterNumber > len(word) / 2:
                            UNKtext.append("UNk")
                        else:
                            UNKtext.append(word)
                    if preid + 1 == id and " ".join(title).strip() != "":
                        AllList.append({'text:': " ".join(UNKtext), 'title:': " ".join(UNKtitle), 'id: ': id})
                    else:
                        print("id" + str(id) + " title: " + title + " text: " + text)
                    preid = id


            except:
                print(preid)
                print(id)
                print("Error occurs!!" + " id: " + str(id))
                #loginformation("", "info", "Error occurs!!" + " id: " + str(id))
                pass
    print(len(AllList))
    #with open(pickleFile, 'wb') as pf:
    #    pickle.dump(AllList, pf, -1)
    return


def UNKDataPickl(sparseWordFile, pickleFile):    #加稀疏词列表
    sparseWordList = []
    with open(sparseWordFile) as swf:
        for line in swf:
            sparseWordList.append(line.split(":")[0])
    #'''
    AllList = []
    with open(pickleFile, 'rb') as pf:
        AllList = pickle.load(pf)
    for i in range(len(AllList)):
        item = AllList[i]
        title = item['title:']
        text = item['text:']
        UNKtitle = []
        for word in title.split(" "):
            if word in sparseWordList:
                UNKtitle.append("UNK")
            else:
                digitOrCharacterNumber = 0
                for index in range(len(word)):
                    # Judge english characters or digits
                    if (word[index] >= u'\u0041' and word[index] <= u'\u005a') \
                            or (word[index] >= u'\u0061' and word[index] <= u'\u007a') \
                            or (word[index] >= u'\u0030' and word[index] <= u'\u0039'):
                        digitOrCharacterNumber += 1
                if digitOrCharacterNumber > len(word) / 2:
                    UNKtitle.append("UNk")
                else:
                    UNKtitle.append(word)
        item['title:'] = " ".join(UNKtitle)
        UNKtext = []
        for word in text.split(" "):
            if word in sparseWordList:
                UNKtext.append("UNK")
            else:
                digitOrCharacterNumber = 0
                for index in range(len(word)):
                    # Judge english characters or digits
                    if (word[index] >= u'\u0041' and word[index] <= u'\u005a') \
                            or (word[index] >= u'\u0061' and word[index] <= u'\u007a') \
                            or (word[index] >= u'\u0030' and word[index] <= u'\u0039'):
                        digitOrCharacterNumber += 1
                if digitOrCharacterNumber > len(word) / 2:
                    UNKtext.append("UNk")
                else:
                    UNKtext.append(word)
        item['text:'] = " ".join(UNKtext)
        AllList[i] = item
        print(AllList[i])
    for line in AllList:
        print(line)
    return AllList
    #'''
def GetAverageTitleLength(pickleFile):
    totalNum = 0
    with open(pickleFile, 'rb') as pf:
        AllList = pickle.load(pf)
        for item in AllList:
            totalNum += len(item['title:'].split(" "))
    print(totalNum/len(AllList))
    return totalNum/len(AllList)

def CreatBatchModel(pickleFile, word2VecModel):
    print("none")
    

if __name__ == "__main__":
    #dataFile = "../../../data/preprocess/PART_I_segment.txt"
    pickleFile = "../../../data/preprocess/pickledata/PART_III_segment.pkl"
    word2VecModel = ""
    #sparseWordFile = "../../../data/preprocess/sparseFile/PART_I_segment_sparsewordfile.txt"
    #storeUNKDataFile(dataFile, pickleFile)
    #UNKDataPickl(sparseWordFile, pickleFile)
    #GetAverageTitleLength(pickleFile)
    #CreatBatchModel(pickleFile)


    '''
    data ="../../../data/preprocess/PART_III_segment.txt"
    with open(data, 'r') as d:
        for line in d:
            print(line)
    '''
