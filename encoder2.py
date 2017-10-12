from read import Reader
from common import unkkey
from common import paddingkey

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

import torch.nn.init as init
import numpy

class Encoder(nn.Module):
    def __init__(self, hyperParams):
        super(Encoder, self).__init__()
        self.hyperParams = hyperParams
        reader = Reader()
        self.extCharEmb, self.extCharDim = reader.load_pretrain(hyperParams.charEmbFile, hyperParams.extCharAlpha, unkkey, paddingkey)
        self.extCharEmb.weight.requires_grad = False

        self.extBiCharEmb, self.extBiCharDim = reader.load_pretrain(hyperParams.bicharEmbFile, hyperParams.extBicharAlpha, unkkey, paddingkey)
        self.extBiCharEmb.weight.requires_grad = False

        self.charEmb = nn.Embedding(hyperParams.charNum, hyperParams.charEmbSize)
        # init.uniform(self.charEmb.weight,
        #              a=-numpy.sqrt(3 /hyperParams.charEmbSize),
        #              b=numpy.sqrt(3 / hyperParams.charEmbSize))
        self.charDim = hyperParams.charEmbSize
        self.charEmb.weight.requires_grad = True
        for idx in range(self.charDim):
            self.charEmb.weight.data[self.hyperParams.charPaddingID][idx] = 0

        self.charTypeEmb = nn.Embedding(hyperParams.charTypeNum, hyperParams.charTypeEmbSize)
        # init.uniform(self.charTypeEmb.weight,
        #              a=-numpy.sqrt(3 / hyperParams.charTypeEmbSize),
        #              b=numpy.sqrt(3 / hyperParams.charTypeEmbSize))
        self.charTypeDim = hyperParams.charTypeEmbSize
        self.charTypeEmb.weight.requires_grad = True
        for idx in range(self.charTypeDim):
            self.charTypeEmb.weight.data[self.hyperParams.charTypePaddingID][idx] = 0

        self.bicharEmb = nn.Embedding(hyperParams.bicharNum, hyperParams.bicharEmbSize)
        # init.uniform(self.bicharEmb.weight,
        #              a=-numpy.sqrt(3 /hyperParams.bicharEmbSize),
        #              b=numpy.sqrt(3 / hyperParams.bicharEmbSize))
        self.bicharDim = hyperParams.bicharEmbSize
        self.bicharEmb.weight.requires_grad = True
        for idx in range(self.bicharDim):
            self.bicharEmb.weight.data[self.hyperParams.bicharPaddingID][idx] = 0


        self.dropOut = nn.Dropout(hyperParams.dropProb)

        self.inputDim = self.extCharDim + self.charDim + self.extBiCharDim + self.bicharDim + self.charTypeDim
        #self.inputDim = self.charDim + self.bicharDim

        self.linearLayer = nn.Linear(in_features= self.inputDim,
                                     out_features=hyperParams.hiddenSize,
                                     bias=True)
        init.xavier_uniform(self.linearLayer.weight)
        self.linearLayer.bias.data.uniform_(-numpy.sqrt(6 / (hyperParams.hiddenSize + 1)),
                                           numpy.sqrt(6 / (hyperParams.hiddenSize + 1)))

        self.lstm_left = nn.LSTMCell(input_size=hyperParams.hiddenSize,
                                     hidden_size=hyperParams.rnnHiddenSize,
                                     bias=True)

        self.lstm_right = nn.LSTMCell(input_size=hyperParams.hiddenSize,
                                      hidden_size=hyperParams.rnnHiddenSize,
                                      bias=True)


        init.xavier_uniform(self.lstm_left.weight_ih)
        init.xavier_uniform(self.lstm_left.weight_hh)
        self.lstm_left.bias_hh.data.uniform_(-numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)),
                                             numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)))
        self.lstm_left.bias_ih.data.uniform_(-numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)),
                                             numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)))

        init.xavier_uniform(self.lstm_right.weight_ih)
        init.xavier_uniform(self.lstm_right.weight_hh)
        self.lstm_right.bias_hh.data.uniform_(-numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)),
                                              numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)))
        self.lstm_right.bias_ih.data.uniform_(-numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)),
                                              numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)))

    def init_cell_hidden(self, batch = 1):
        if self.hyperParams.useCuda:
            return (torch.autograd.Variable(torch.zeros(batch, self.hyperParams.rnnHiddenSize)).cuda(self.hyperParams.gpuID),
                    torch.autograd.Variable(torch.zeros(batch, self.hyperParams.rnnHiddenSize)).cuda(self.hyperParams.gpuID))
        else:
            return (torch.autograd.Variable(torch.zeros(batch, self.hyperParams.rnnHiddenSize)),
                    torch.autograd.Variable(torch.zeros(batch, self.hyperParams.rnnHiddenSize)))

    def forward(self, feats):
        batch = feats.batch
        charTypeIndexes = feats.char_type_feats
        charIndexes = feats.char_feats
        leftbicharIndexes = feats.leftbichar_feats
        rightbicharIndexes = feats.rightbichar_feats
        extCharIndexes = feats.extchar_feats
        leftextbicharIndexes = feats.leftextbichar_feats
        rightextbicharIndexes = feats.rightextbichar_feats
        charType = self.charTypeEmb(charTypeIndexes)
        extChar = self.extCharEmb(extCharIndexes)
        char = self.charEmb(charIndexes)

        leftBichar = self.bicharEmb(leftbicharIndexes)
        rightBichar = self.bicharEmb(rightbicharIndexes)
        leftExtBichar = self.extBiCharEmb(leftextbicharIndexes)
        rightExtBichar = self.extBiCharEmb(rightextbicharIndexes)

        char_num = extChar.size()[1]

        charType = self.dropOut(charType)
        extChar = self.dropOut(extChar)
        char = self.dropOut(char)

        leftBichar = self.dropOut(leftBichar)
        rightBichar = self.dropOut(rightBichar)
        leftExtBichar = self.dropOut(leftExtBichar)
        rightExtBichar = self.dropOut(rightExtBichar)

        leftConcat = torch.cat((char, extChar, leftBichar, leftExtBichar, charType), 2)
        leftConcat = leftConcat.view(batch * char_num, self.inputDim)

        rightConcat = torch.cat((char, extChar, rightBichar, rightExtBichar, charType), 2)
        rightConcat = rightConcat.view(batch * char_num, self.inputDim)
        leftNoLinear = self.dropOut(F.tanh(self.linearLayer(leftConcat)))
        leftNoLinear = leftNoLinear.view(batch, char_num, self.hyperParams.hiddenSize)
        leftLSTMinput = leftNoLinear.permute(1, 0, 2)

        left_h, left_c = self.init_cell_hidden(batch)
        leftLSTMoutput = []
        for idx in range(char_num):
            left_h, left_c = self.lstm_left(leftLSTMinput[idx], (left_h, left_c))
            left_h = self.dropOut(left_h)
            leftLSTMoutput.append(left_h.view(batch, 1, self.hyperParams.rnnHiddenSize))
        leftLSTMoutput = torch.cat(leftLSTMoutput, 1)

        rightNoLinear = self.dropOut(F.tanh(self.linearLayer(rightConcat)))
        rightNoLinear = rightNoLinear.view(batch, char_num, self.hyperParams.hiddenSize)
        rightLSTMinput = rightNoLinear.permute(1, 0, 2)

        right_h , right_c = self.init_cell_hidden(batch)
        rightLSTMoutput = []
        for idx in reversed(range(char_num)):
            right_h, right_c = self.lstm_right(rightLSTMinput[idx], (right_h, right_c))
            right_h = self.dropOut(right_h)
            rightLSTMoutput.insert(0, right_h.view(batch, 1, self.hyperParams.rnnHiddenSize))
        rightLSTMoutput = torch.cat(rightLSTMoutput, 1)

        output = torch.cat((leftLSTMoutput, rightLSTMoutput), 2)
        return output

