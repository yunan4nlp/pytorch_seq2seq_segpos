from instance import  Instance
import unicodedata
import torch
import re
import torch.nn as nn
import torch.nn.init as init
import numpy

class Reader:

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def readInstances(self, path, maxInst = -1):
        insts = []
        r = open(path, encoding='utf8')
        for line in r.readlines():
            line = unicodedata.normalize('NFKC',line.strip())
            inst = Instance()
            if line != "":
                info = line.split(" ")
                count = 0
                for data in info:
                    pos = data.find('_')
                    word = data[0:pos]
                    label = data[pos + 1:]
                    inst.m_words.append(word)
                    word_len = len(word)
                    inst.m_gold_str.append('[' + str(count) + ',' + str(count + word_len) + ']')
                    count += word_len
                    for idx in range(word_len):
                        inst.m_chars.append(word[idx])
                        if idx == 0:
                            inst.m_gold.append('SEP#' + label)
                            inst.m_pos.append(label)
                        else:
                            inst.m_gold.append('APP')
                for idx in range(len(inst.m_chars)):
                    if idx - 1 >= 0:
                        inst.m_bichars.append(inst.m_chars[idx - 1] + inst.m_chars[idx])
                    else:
                        inst.m_bichars.append('<s>' + inst.m_chars[idx])
                if len(insts) == maxInst:
                    break
                inst.m_char_size = len(inst.m_chars)
                inst.m_word_size = len(inst.m_words)
                inst.m_bichar_size = len(inst.m_bichars)
                inst.m_gold_size = len(inst.m_gold)
                insts.append(inst)
            else:
                break
        r.close()
        return insts

    def load_pretrain(self, file, alpha, unk):
        f = open(file, encoding='utf-8')
        allLines = f.readlines()
        indexs = []
        info = allLines[0].strip().split(' ')
        embDim = len(info) - 1
        emb = nn.Embedding(alpha.m_size, embDim)
        init.xavier_uniform(emb.weight)
        oov_emb = torch.zeros(1, embDim).type(torch.FloatTensor)
        for line in allLines:
            info = line.strip().split(' ')
            wordID = alpha.from_string(info[0])
            if wordID >= 0:
                indexs.append(wordID)
                for idx in range(embDim):
                    val = float(info[idx + 1])
                    emb.weight.data[wordID][idx] = val
                    oov_emb[0][idx] += val
        f.close()
        count = len(indexs)
        for idx in range(embDim):
            oov_emb[0][idx] /= count
        unkID = alpha.from_string(unk)
        print('UNK ID: ', unkID)
        if unkID != -1:
            for idx in range(embDim):
                emb.weight.data[unkID][idx] = oov_emb[0][idx]
        print("Load Embedding file: ", file, ", size: ", embDim)
        oov = 0
        for idx in range(alpha.m_size):
            if idx not in indexs:
                oov += 1
        print("OOV Num: ", oov, "Total Num: ", alpha.m_size,
              "OOV Ratio: ", oov / alpha.m_size)
        print("OOV ", unk, "use avg value initialize")
        return emb, embDim

