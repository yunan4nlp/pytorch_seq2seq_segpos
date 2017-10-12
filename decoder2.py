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
        self.wordLenEmb = nn.Embedding(hyperParams.wordLenNum, hyperParams.wordLenEmbSize)
        init.uniform(self.wordLenEmb.weight,
                     a=-numpy.sqrt(3 / hyperParams.wordLenEmbSize),
                     b=numpy.sqrt(3 / hyperParams.wordLenEmbSize))
        self.wordLemDim = hyperParams.wordLenEmbSize
        self.wordLenEmb.weight.requires_grad = True

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
        self.linearLayer = nn.Linear(in_features=hyperParams.rnnHiddenSize * 2 + hyperParams.hiddenSize,
                                     out_features=hyperParams.labelSize,
                                     bias=False)

        self.combineWordPos = nn.Linear(in_features=hyperParams.rnnHiddenSize * 2 + hyperParams.posEmbSize + hyperParams.wordLenEmbSize,
                                        out_features=hyperParams.hiddenSize,
                                        bias=True)

        #self.linearLayer.weight.data.uniform_(-0.01, 0.01)
        init.xavier_uniform(self.linearLayer.weight)
        init.xavier_uniform(self.combineWordPos.weight)
        self.combineWordPos.bias.data.uniform_(-numpy.sqrt(6 / (hyperParams.hiddenSize + 1)),
                                               numpy.sqrt(6 / (hyperParams.hiddenSize + 1)))

        self.dropOut = nn.Dropout(hyperParams.dropProb)
        self.softmax = nn.LogSoftmax()

        self.z_bucket = torch.autograd.Variable(torch.zeros(1, hyperParams.hiddenSize)).type(torch.FloatTensor)
        self.h_bucket = torch.autograd.Variable(torch.zeros(1, hyperParams.rnnHiddenSize)).type(torch.FloatTensor)
        self.c_bucket = torch.autograd.Variable(torch.zeros(1, hyperParams.rnnHiddenSize)).type(torch.FloatTensor)
        if hyperParams.useCuda:
            self.z_bucket = self.z_bucket.cuda(self.hyperParams.gpuID)
            self.h_bucket = self.h_bucket.cuda(self.hyperParams.gpuID)
            self.c_bucket = self.c_bucket.cuda(self.hyperParams.gpuID)


    def forward(self, insts, encoder_output, batch = 1, bTrain = False):
        char_num = encoder_output.size()[1]
        batch_output = []
        batch_state = []
        for idx in range(batch):
            inst = insts[idx]
            s = state(inst, self.hyperParams)
            #s.h, s.c = self.incLSTM(s.z, (s.h, s.c))
            #s.h = self.dropOut(s.h)
            sent_output = []
            real_char_num = inst.m_char_size
            for idy in range(char_num):
                if idy < real_char_num:
                    h_now, c_now = self.prepare(s, idy, encoder_output[idx])
                    #print(encoder_output[idx][idy].view(1, self.hyperParams.rnnHiddenSize * 2))
                    v = torch.cat((h_now, encoder_output[idx][idy].view(1, self.hyperParams.rnnHiddenSize * 2)), 1)
                    #v = torch.cat((self.bucket_rnn, encoder_output[idx][idy].view(1, self.hyperParams.rnnHiddenSize * 2)), 1)
                    #v = encoder_output[idx][idy].view(1, self.hyperParams.rnnHiddenSize * 2)
                    output = self.linearLayer(v)
                    if idy == 0:
                        output.data[0][self.hyperParams.appID] = -1e+99
                    #self.action(s, idy, encoder_output[idx], output, bTrain)
                    self.my_action(s, idy, output, h_now, c_now, bTrain)
                    #output = self.softmax(output)
                    sent_output.append(output)
                else:
                    sent_output.append(self.bucket)
            sent_output = torch.cat(sent_output, 0)
            batch_output.append(sent_output)
            #print(s.actions)
            batch_state.append(s)
        batch_output = torch.cat(batch_output, 0)
        batch_output = self.softmax(batch_output)
        #encoder_output = torch.cat(encoder_output, 0)
        #output = self.softmax(self.linearLayer(encoder_output))
        return batch_output, batch_state

    def prepare(self, state, index, encoder_char):
        if index == 0:
            h_last = self.h_bucket
            c_last = self.c_bucket
            z = self.z_bucket
        else:
            h_last = state.word_hiddens[-1]
            c_last = state.word_cells[-1]
            if len(state.pos_id) >= 1:
                last_pos = torch.autograd.Variable(torch.zeros(1)).type(torch.LongTensor)
                if self.hyperParams.useCuda: last_pos = last_pos.cuda(self.hyperParams.gpuID)
                last_pos.data[0] = state.pos_id[-1]
                last_pos_emb = self.dropOut(self.posEmb(last_pos))
            if len(state.words) >= 1:
                last_word_len = len(state.words[-1])
                start = index - last_word_len
                end = index
                chars_emb = []
                #print(start, ",", end, ",", state.pos_labels[-1])
                for idx in range(start, end):
                    chars_emb.append(encoder_char[idx].view(1, 1, 2 * self.hyperParams.rnnHiddenSize))
                chars_emb = torch.cat(chars_emb, 1)
                last_word_emb = F.avg_pool1d(chars_emb.permute(0, 2, 1), last_word_len).view(1, self.hyperParams.rnnHiddenSize * 2)
            if last_word_len > 6:last_word_len = 6
            word_len_id = self.hyperParams.wordLenAlpha.from_string(str(last_word_len))
            word_len = torch.autograd.Variable(torch.zeros(1)).type(torch.LongTensor)
            word_len.data[0] = word_len_id
            if self.hyperParams.useCuda: word_len = word_len.cuda(self.hyperParams.gpuID)
            word_len_emb = self.wordLenEmb(word_len)
            concat = torch.cat((last_pos_emb, last_word_emb, word_len_emb), 1)
            z = self.dropOut(F.tanh(self.combineWordPos(concat)))
        #print(h_last.data[0][0])
        h_now, c_now = self.incLSTM(z, (h_last, c_last))
        state.all_h.append(h_now)
        state.all_c.append(c_now)
        return h_now, c_now

    def my_action(self, state, index, output, h_now, c_now, bTrain):
        if bTrain:
            action = state.m_gold[index]
        else:
            actionID = getMaxIndex(self.hyperParams, output.view(self.hyperParams.labelSize))
            action = self.hyperParams.labelAlpha.from_id(actionID)
        state.actions.append(action)

        pos = action.find('#')
        if pos == -1:
            ###app
            state.words[-1] += state.m_chars[index]
        else:
            ###sep
            tmp_word = state.m_chars[index]
            state.words.append(tmp_word)
            posLabel = action[pos + 1:]
            state.pos_labels.append(posLabel)
            posID = self.hyperParams.posAlpha.from_string(posLabel)
            state.pos_id.append(posID)
            state.word_cells.append(c_now)
            state.word_hiddens.append(h_now)

            # def action(self, state, index, encoder_char, output, bTrain):
            #     if bTrain:
            #         action = state.m_gold[index]
            #     else:
            #         actionID = getMaxIndex(self.hyperParams, output.view(self.hyperParams.labelSize))
            #         action = self.hyperParams.labelAlpha.from_id(actionID)
            #     state.actions.append(action)
            #     if len(state.pos_labels) >= 1:
            #         state.last_pos.data[0] = state.pos_id[-1]
            #         state.last_pos_emb = self.dropOut(self.posEmb(state.last_pos))
            #
            #     if len(state.words) >= 1:
            #         last_word_len = len(state.words[-1])
            #         start = index - last_word_len
            #         end = index
            #         chars_emb = []
            #         for idx in range(start, end):
            #             chars_emb.append(encoder_char[idx].view(1, 1, 2 * self.hyperParams.rnnHiddenSize))
            #         chars_emb = torch.cat(chars_emb, 1)
            #         state.last_word_emb = F.avg_pool1d(chars_emb.permute(0, 2, 1), last_word_len).view(1, self.hyperParams.rnnHiddenSize * 2)
            #
            #         concat = torch.cat((state.last_pos_emb, state.last_word_emb), 1)
            #         state.z = self.dropOut(F.tanh(self.combineWordPos(concat)))
            #         state.h, state.c = self.incLSTM(state.z, (state.word_hiddens[-1], state.word_cells[-1]))
            #
            #         state.h = self.dropOut(state.h)
            #
            #     pos = action.find('#')
            #     if pos == -1:
            #         ###app
            #         state.words[-1] += state.m_chars[index]
            #     else:
            #         ###sep
            #         tmp_word = state.m_chars[index]
            #         state.words.append(tmp_word)
            #         posLabel = action[pos + 1:]
            #         state.pos_labels.append(posLabel)
            #         posID = self.hyperParams.posAlpha.from_string(posLabel)
            #         state.pos_id.append(posID)
            #
            #         state.word_cells.append(state.c)
            #         state.word_hiddens.append(state.h)
            #     print(state.words, state.pos_labels)
            #

