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
                     a=-numpy.sqrt(3 / hyperParams.charEmbSize),
                     b=numpy.sqrt(3 / hyperParams.charEmbSize))
        self.charDim = hyperParams.charEmbSize
        self.charEmb.weight.requires_grad = True

        self.bicharEmb = nn.Embedding(hyperParams.bicharNum, hyperParams.bicharEmbSize)
        init.uniform(self.bicharEmb.weight,
                     a=-numpy.sqrt(3 / hyperParams.bicharEmbSize),
                     b=numpy.sqrt(3 / hyperParams.bicharEmbSize))
        self.bicharDim = hyperParams.bicharEmbSize
        self.bicharEmb.weight.requires_grad = True


        self.dropOut = nn.Dropout(hyperParams.dropProb)

        self.inputDim = self.extCharDim + self.charDim + self.extBiCharDim + self.bicharDim
        #self.inputDim = self.charDim + self.bicharDim
        self.linearLayer = nn.Linear(in_features= self.inputDim,
                                     out_features=hyperParams.hiddenSize,
                                     bias=True)
        init.xavier_uniform(self.linearLayer.weight)

        self.bilstm = nn.LSTM(input_size=hyperParams.hiddenSize,
                              hidden_size=hyperParams.rnnHiddenSize,
                              batch_first=True,
                              bidirectional=True,
                              bias=True,
                              dropout=hyperParams.dropProb)

    def init_hidden(self, batch = 1):
        if self.hyperParams.useCuda:
            return (torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)).cuda(),
                    torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)).cuda())
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

        concat = torch.cat((char, extChar, biChar, extBiChar), 2)
        #concat = torch.cat((char,  biChar), 2)
        concat = concat.view(batch * char_num, self.inputDim)
        nonlinearOutput = self.dropOut(F.tanh(self.linearLayer(concat)))
        nonlinearOutput = nonlinearOutput.view(batch, char_num, self.hyperParams.hiddenSize)
        output, hidden = self.bilstm(nonlinearOutput, hidden)
        return output, hidden

