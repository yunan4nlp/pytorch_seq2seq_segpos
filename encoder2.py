from read import Reader

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
        self.extCharEmb, self.extCharDim = reader.load_pretrain(hyperParams.charEmbFile, hyperParams.charAlpha, hyperParams.unk)
        self.extCharEmb.weight.requires_grad = False

        self.extBiCharEmb, self.extBiCharDim = reader.load_pretrain(hyperParams.bicharEmbFile, hyperParams.bicharAlpha, hyperParams.unk)
        self.extBiCharEmb.weight.requires_grad = False

        self.charEmb = nn.Embedding(hyperParams.charNum, hyperParams.charEmbSize)
        init.uniform(self.charEmb.weight,
                     a=-numpy.sqrt(3 /hyperParams.charEmbSize),
                     b=numpy.sqrt(3 / hyperParams.charEmbSize))

        self.charDim = hyperParams.charEmbSize
        self.charEmb.weight.requires_grad = True

        self.bicharEmb = nn.Embedding(hyperParams.bicharNum, hyperParams.bicharEmbSize)
        init.uniform(self.bicharEmb.weight,
                     a=-numpy.sqrt(3 /hyperParams.bicharEmbSize),
                     b=numpy.sqrt(3 / hyperParams.bicharEmbSize))

        self.bicharDim = hyperParams.bicharEmbSize
        self.bicharEmb.weight.requires_grad = True


        self.dropOut = nn.Dropout(hyperParams.dropProb)

        self.inputDim = self.extCharDim + self.charDim + self.extBiCharDim + self.bicharDim
        #self.inputDim = self.charDim + self.bicharDim

        self.leftLayer = nn.Linear(in_features= self.inputDim,
                                   out_features=hyperParams.hiddenSize,
                                   bias=True)
        init.xavier_uniform(self.leftLayer.weight)
        self.leftLayer.bias.data.uniform_(-numpy.sqrt(6 / (hyperParams.hiddenSize + 1)),
                                          numpy.sqrt(6 / (hyperParams.hiddenSize + 1)))

        self.rightLayer = nn.Linear(in_features= self.inputDim,
                                    out_features=hyperParams.hiddenSize,
                                    bias=True)
        init.xavier_uniform(self.rightLayer.weight)
        self.rightLayer.bias.data.uniform_(-numpy.sqrt(6 / (hyperParams.hiddenSize + 1)),
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

    def init_bucket_extbichar(self, batch = 1):
        if self.hyperParams.useCuda:
            return (torch.autograd.Variable(torch.zeros(batch, 1, self.extBiCharDim)).cuda(self.hyperParams.gpuID))
        else:
            return (torch.autograd.Variable(torch.zeros(batch, 1, self.extBiCharDim)))

    def init_bucket_bichar(self, batch = 1):
        if self.hyperParams.useCuda:
            return (torch.autograd.Variable(torch.zeros(batch, 1, self.hyperParams.bicharEmbSize)).cuda(self.hyperParams.gpuID))
        else:
            return (torch.autograd.Variable(torch.zeros(batch, 1, self.hyperParams.bicharEmbSize)))

    def init_cell_hidden(self, batch = 1):
        if self.hyperParams.useCuda:
            return (torch.autograd.Variable(torch.zeros(batch, self.hyperParams.rnnHiddenSize)).cuda(self.hyperParams.gpuID),
                    torch.autograd.Variable(torch.zeros(batch, self.hyperParams.rnnHiddenSize)).cuda(self.hyperParams.gpuID))
        else:
            return (torch.autograd.Variable(torch.zeros(batch, self.hyperParams.rnnHiddenSize)),
                    torch.autograd.Variable(torch.zeros(batch, self.hyperParams.rnnHiddenSize)))

    def init_hidden(self, batch = 1):
        if self.hyperParams.useCuda:
            return (torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)).cuda(self.hyperParams.gpuID),
                    torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)).cuda(self.hyperParams.gpuID))
        else:
            return (torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)),
                    torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)))

    def forward(self, charIndexes, bicharIndexes, hidden, batch = 1):
        extChar = self.extCharEmb(charIndexes)
        char = self.charEmb(charIndexes)
        extBiChar = self.extBiCharEmb(bicharIndexes)
        biChar = self.bicharEmb(bicharIndexes)

        char_num = extChar.size()[1]

        extChar = self.dropOut(extChar)
        char = self.dropOut(char)
        extBiChar = self.dropOut(extBiChar)
        biChar = self.dropOut(biChar)

        bucketExtbichar = self.init_bucket_extbichar(batch)
        if char_num > 1:
            leftExtBichar = torch.cat(torch.split(extBiChar, 1, 1)[1:], 1)
            leftExtBichar = torch.cat((leftExtBichar, bucketExtbichar), 1)
        else:
            leftExtBichar = bucketExtbichar

        bucketBichar = self.init_bucket_bichar(batch)
        if char_num > 1:
            leftBichar = torch.cat(torch.split(biChar, 1, 1)[1:], 1)
            leftBichar = torch.cat((leftBichar, bucketBichar), 1)
        else:
            leftBichar = bucketBichar
        leftConcat = torch.cat((char, extChar, leftBichar, leftExtBichar), 2)
        leftConcat = leftConcat.view(batch * char_num, self.inputDim)

        rightConcat = torch.cat((char, extChar, biChar, extBiChar), 2)
        #concat = torch.cat((char,  biChar), 2)
        rightConcat = rightConcat.view(batch * char_num, self.inputDim)

        leftNoLinear = self.dropOut(F.tanh(self.leftLayer(leftConcat)))
        leftNoLinear = leftNoLinear.view(batch, char_num, self.hyperParams.hiddenSize)
        leftLSTMinput = leftNoLinear.permute(1, 0, 2)

        left_h, left_c = self.init_cell_hidden(batch)
        leftLSTMoutput = []
        for idx in range(char_num):
            left_h, left_c = self.lstm_left(leftLSTMinput[idx], (left_h, left_c))
            left_h = self.dropOut(left_h)
            leftLSTMoutput.append(left_h.view(1, batch, self.hyperParams.rnnHiddenSize))
        leftLSTMoutput = torch.cat(leftLSTMoutput, 0).permute(1, 0, 2)

        rightNoLinear = self.dropOut(F.tanh(self.rightLayer(rightConcat)))
        rightNoLinear = rightNoLinear.view(batch, char_num, self.hyperParams.hiddenSize)
        rightLSTMinput = rightNoLinear.permute(1, 0, 2)

        right_h , right_c = self.init_cell_hidden(batch)
        rightLSTMoutput = []
        for idx in reversed(range(char_num)):
            right_h, right_c = self.lstm_right(rightLSTMinput[idx], (right_h, right_c))
            right_h = self.dropOut(right_h)
            rightLSTMoutput.insert(0, right_h.view(1, batch, self.hyperParams.rnnHiddenSize))
        rightLSTMoutput = torch.cat(rightLSTMoutput, 0).permute(1, 0, 2)

        output = torch.cat((leftLSTMoutput, rightLSTMoutput), 2)
        return output, hidden

