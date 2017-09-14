import torch.nn as nn
from read import Reader
import torch.autograd

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
        self.charDim = hyperParams.charEmbSize
        self.charEmb.weight.requires_grad = True

        self.bicharEmb = nn.Embedding(hyperParams.bicharNum, hyperParams.bicharEmbSize)
        self.bicharDim = hyperParams.bicharEmbSize
        self.bicharEmb.weight.requires_grad = True

        self.dropOut = nn.Dropout(hyperParams.dropProb)

        self.inputDim = self.extCharDim + self.charDim + self.extBiCharDim + self.bicharDim
        self.linearLayer = nn.Linear(in_features= self.inputDim,
                                     out_features=hyperParams.hiddenSize)

        self.bilstm = nn.LSTM(input_size=hyperParams.hiddenSize,
                          hidden_size=hyperParams.rnnHiddenSize,
                          batch_first=True,
                          bidirectional=True,
                          dropout=hyperParams.dropProb)



    def init_hidden(self, batch = 1):
        return (torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)),
                torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)))

    def forward(self, charIndexes, bicharIndexes, hidden, batch = 1):
        extChar = self.extCharEmb(charIndexes)
        char_num = extChar.size()[1]
        char = self.charEmb(charIndexes)
        extBiChar = self.extBiCharEmb(bicharIndexes)
        biChar = self.bicharEmb(bicharIndexes)

        extChar = self.dropOut(extChar)
        char = self.dropOut(char)
        extBiChar = self.dropOut(extBiChar)
        biChar = self.dropOut(biChar)

        concat = torch.cat((char, extChar, biChar, extBiChar), 2)
        concat = concat.view(batch * char_num, self.inputDim)
        linearOutput = self.dropOut(torch.nn.functional.tanh(self.linearLayer(concat)))
        linearOutput = linearOutput.view(batch, char_num, self.hyperParams.hiddenSize)
        output, hidden = self.bilstm(linearOutput, hidden)
        return output, hidden

