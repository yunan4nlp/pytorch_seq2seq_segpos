from common import getMaxIndex
import torch

class state:
    def __init__(self, inst, hyperParams):
        self.hyperParams = hyperParams
        self.m_chars = inst.m_chars
        self.m_gold = inst.m_gold
        self.words = []
        self.pos_id = []
        self.pos_labels = []
        self.actions = []

        self.last_pos = torch.autograd.Variable(torch.zeros(1)).type(torch.LongTensor)
        self.last_word_emb = torch.autograd.Variable(torch.zeros(1, hyperParams.rnnHiddenSize * 2)).type(torch.FloatTensor)
        self.last_pos_emb = torch.autograd.Variable(torch.zeros(1, hyperParams.posEmbSize)).type(torch.FloatTensor)
        self.last_word_pos_emb = torch.autograd.Variable(torch.zeros(1, hyperParams.hiddenSize)).type(torch.FloatTensor)
        self.h = torch.autograd.Variable(torch.zeros(1, hyperParams.rnnHiddenSize)).type(torch.FloatTensor)
        self.c = torch.autograd.Variable(torch.zeros(1, hyperParams.rnnHiddenSize)).type(torch.FloatTensor)

        self.word_start = 0

        if hyperParams.useCuda:
            self.last_pos = self.last_pos.cuda(hyperParams.gpuID)
            self.last_word_emb = self.last_word_emb.cuda(hyperParams.gpuID)
            self.last_pos_emb = self.last_pos_emb.cuda(hyperParams.gpuID)
            self.last_word_pos_emb = self.last_word_pos_emb.cuda(hyperParams.gpuID)
            self.h = self.h.cuda(hyperParams.gpuID)
            self.c = self.c.cuda(hyperParams.gpuID)

