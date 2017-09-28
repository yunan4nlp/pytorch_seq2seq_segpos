import torch
import torch.nn as nn
import torch.nn.functional as F
from state import state
from common import getMaxIndex
import torch.nn.init as init
import numpy


class Decoder(nn.Module):
    def __init__(self, hyperParams):
        super(Decoder, self).__init__()
        self.hyperParams = hyperParams
        self.posEmb = nn.Embedding(hyperParams.posNum, hyperParams.posEmbSize)
        init.uniform(self.posEmb.weight,
                     a=-numpy.sqrt(3 / hyperParams.posEmbSize),
                     b=numpy.sqrt(3 / hyperParams.posEmbSize))

        self.posDim = hyperParams.posEmbSize
        self.posEmb.weight.requires_grad = True
        self.incLSTM = nn.LSTMCell(input_size=hyperParams.hiddenSize,
                                   hidden_size=hyperParams.rnnHiddenSize,
                                   bias=True)
        init.xavier_uniform(self.incLSTM.weight_ih)
        init.xavier_uniform(self.incLSTM.weight_hh)
        self.incLSTM.bias_hh.data.uniform_(-numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)),
                                            numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)))
        self.incLSTM.bias_ih.data.uniform_(-numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)),
                                           numpy.sqrt(6 / (hyperParams.rnnHiddenSize + 1)))

        self.bucket = torch.autograd.Variable(torch.zeros(1, hyperParams.labelSize)).type(torch.FloatTensor)
        if hyperParams.useCuda:self.bucket = self.bucket.cuda(self.hyperParams.gpuID)
        self.bucket_rnn = torch.autograd.Variable(torch.zeros(1, hyperParams.rnnHiddenSize)).type(torch.FloatTensor)
        if hyperParams.useCuda:self.bucket_rnn = self.bucket_rnn.cuda(self.hyperParams.gpuID)
        self.linearLayer = nn.Linear(in_features=hyperParams.rnnHiddenSize * 2,
                                     out_features=hyperParams.labelSize,
                                     bias=False)
        self.combineWordPos = nn.Linear(in_features=hyperParams.rnnHiddenSize * 2 + hyperParams.posEmbSize,
                                        out_features=hyperParams.hiddenSize,
                                        bias=True)

        init.xavier_uniform(self.linearLayer.weight)
        init.xavier_uniform(self.combineWordPos.weight)
        self.combineWordPos.bias.data.uniform_(-numpy.sqrt(6 / (hyperParams.hiddenSize + 1)),
                                            numpy.sqrt(6 / (hyperParams.hiddenSize + 1)))

        self.dropOut = nn.Dropout(hyperParams.dropProb)
        self.softmax = nn.LogSoftmax()


    def forward(self, insts, encoder_output, batch = 1, bTrain = False):
        char_num = encoder_output.size()[1]
        batch_output = []
        batch_state = []
        for idx in range(batch):
            inst = insts[idx]
            s = state(inst, self.hyperParams)
            s.h, s.c = self.incLSTM(s.last_word_pos_emb, (s.h, s.c))
            s.h = self.dropOut(s.h)
            sent_output = []
            real_char_num = inst.m_char_size
            for idy in range(char_num):
                if idy < real_char_num:
                    #print(encoder_output[idx][idy].view(1, self.hyperParams.rnnHiddenSize * 2))
                    #v = torch.cat((s.h, encoder_output[idx][idy].view(1, self.hyperParams.rnnHiddenSize * 2)), 1)
                    #v = torch.cat((self.bucket_rnn, encoder_output[idx][idy].view(1, self.hyperParams.rnnHiddenSize * 2)), 1)
                    v = encoder_output[idx][idy].view(1, self.hyperParams.rnnHiddenSize * 2)
                    output = self.linearLayer(v)
                    self.action(s, idy, encoder_output[idx], output, bTrain)
                    sent_output.append(output)
                else:
                    sent_output.append(self.bucket)
            sent_output = torch.cat(sent_output, 0)
            batch_output.append(sent_output)
            batch_state.append(s)
        batch_output = torch.cat(batch_output, 0)
        batch_output = self.softmax(batch_output)
        #encoder_output = torch.cat(encoder_output, 0)
        #output = self.softmax(self.linearLayer(encoder_output))
        return batch_output, batch_state

    def action(self, state, index, encoder_char, output, bTrain):
        if bTrain:
            action = state.m_gold[index]
        else:
            actionID = getMaxIndex(self.hyperParams, output.view(self.hyperParams.labelSize))
            action = self.hyperParams.labelAlpha.from_id(actionID)
        state.actions.append(action)
        if len(state.pos_labels) >= 1:
            state.last_pos.data[0] = state.pos_id[-1]
            state.last_pos_emb = self.dropOut(self.posEmb(state.last_pos))

        if len(state.words) >= 1:
            last_word_len = len(state.words[-1])
            start = index - last_word_len
            end = index
            chars_emb = []
            for idx in range(start, end):
                chars_emb.append(encoder_char[idx].view(1, 1, 2 * self.hyperParams.rnnHiddenSize))
            chars_emb = torch.cat(chars_emb, 1)
            state.last_word_emb = F.avg_pool1d(chars_emb.permute(0, 2, 1), last_word_len).view(1, self.hyperParams.rnnHiddenSize * 2)

            concat = torch.cat((state.last_pos_emb, state.last_word_emb), 1)
            state.last_word_pos_emb = self.dropOut(F.tanh(self.combineWordPos(concat)))
            state.h, state.c = self.incLSTM(state.last_word_pos_emb, (state.h, state.c))
            state.h = self.dropOut(state.h)

        pos = action.find('#')
        if pos == -1:
            ###app
            if len(state.words) == 0:
                state.words.append("")
                state.pos_id.append(self.hyperParams.posUNKID)
                state.pos_labels.append(self.hyperParams.posAlpha.from_id(self.hyperParams.posUNKID))
                state.words[-1] += state.m_chars[index]
            else:
                state.words[-1] += state.m_chars[index]
        else:
            ###sep
            tmp_word = state.m_chars[index]
            state.words.append(tmp_word)
            posLabel = action[pos + 1:]
            state.pos_labels.append(posLabel)
            posID = self.hyperParams.posAlpha.from_string(posLabel)
            state.pos_id.append(posID)


