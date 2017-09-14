from common import getMaxIndex
import torch

class state:
    def __init__(self, inst, hyperParams):
        self.hyperParams = hyperParams
        self.m_chars = inst.m_chars
        self.words = []
        self.pos_id = []
        self.actions = []

        self.last_pos = torch.autograd.Variable(torch.zeros(1)).type(torch.LongTensor)

        self.last_word_emb = torch.autograd.Variable(torch.zeros(1, hyperParams.charEmbSize)).type(torch.FloatTensor)
        self.last_pos_emb = torch.autograd.Variable(torch.zeros(1, hyperParams.posEmbSize)).type(torch.FloatTensor)

        self.last_word_pos_emb = torch.autograd.Variable(torch.zeros(1, hyperParams.charEmbSize + hyperParams.posEmbSize)).type(torch.FloatTensor)
        self.h = torch.autograd.Variable(torch.zeros(1, hyperParams.rnnHiddenSize)).type(torch.FloatTensor)
        self.c = torch.autograd.Variable(torch.zeros(1, hyperParams.rnnHiddenSize)).type(torch.FloatTensor)

