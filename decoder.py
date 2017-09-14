import torch.nn as nn
import torch
from state import state
from common import getMaxIndex

class Decoder(nn.Module):
    def __init__(self, hyperParams):
        super(Decoder, self).__init__()
        self.hyperParams = hyperParams
        self.lastWordHidden = []
        self.posEmb = nn.Embedding(hyperParams.posNum, hyperParams.posEmbSize)
        self.posDim = hyperParams.posEmbSize
        self.posEmb.weight.requires_grad = True
        self.incLSTM = nn.LSTMCell(input_size=hyperParams.charEmbSize + hyperParams.posEmbSize,
                                   hidden_size=hyperParams.rnnHiddenSize,
                                   bias=True)
        self.bucket = torch.autograd.Variable(torch.zeros(1, hyperParams.labelSize)).type(torch.FloatTensor)
        self.linearLayer = nn.Linear(hyperParams.rnnHiddenSize + hyperParams.rnnHiddenSize * 2, hyperParams.labelSize)
        self.dropOut = nn.Dropout(hyperParams.dropProb)
        self.softmax = nn.LogSoftmax()


    def forward(self, insts, encoder_output, batch = 1):
        char_num = encoder_output.size()[1]
        batch_output = []
        batch_state = []
        for idx in range(batch):
            s = state(insts[idx], self.hyperParams)
            s.h, s.c = self.incLSTM(s.last_word_pos_emb, (s.h, s.c))
            sent_output = []
            real_char_num = insts[idx].m_char_size
            for idy in range(char_num):
                if idy < real_char_num:
                    v = torch.cat((self.dropOut(s.h), encoder_output[idx][idy].view(1, self.hyperParams.rnnHiddenSize * 2)), 1)
                    output = self.dropOut(torch.nn.functional.tanh(self.linearLayer(v)))
                    self.action(s, idy, encoder_output[idx], output)
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

    def action(self, state, index, encoder_char, output):
        actionID = getMaxIndex(self.hyperParams, output.view(self.hyperParams.labelSize))
        action = self.hyperParams.labelAlpha.from_id(actionID)
        state.actions.append(action)
        pos = action.find('#')
        if pos == -1:
            if len(state.words) == 0:
                state.words.append("")
                state.words[-1] += state.m_chars[index]
            else:
                state.words[-1] += state.m_chars[index]
        else:
            tmp_word = state.m_chars[index]
            state.words.append(tmp_word)
            posLabel = action[pos + 1:]
            posID = self.hyperParams.posAlpha.from_string(posLabel)
            state.pos_id.append(posID)

            if len(state.pos_id) >= 2:
                state.last_pos.data[0] = state.pos_id[-2]
                state.last_pos_emb = self.posEmb(state.last_pos)
            if len(state.words) >= 2:
                last_word_len = len(state.words[-2])
                start = index - len(state.words[-1]) - last_word_len + 1
                end = start + last_word_len
                chars_emb = []
                for idx in range(start, end):
                    chars_emb.append(encoder_char[idx].view(1, 1, self.hyperParams.charEmbSize))
                chars_emb = torch.cat(chars_emb, 1)
                state.last_word_emb = torch.nn.functional.avg_pool1d(chars_emb.permute(0, 2, 1), last_word_len).view(1, self.hyperParams.charEmbSize)
                state.last_word_pos_emb = torch.cat((state.last_pos_emb, state.last_word_emb), 1)
                state.h, state.c = self.incLSTM(state.last_word_pos_emb, (state.h, state.c))
