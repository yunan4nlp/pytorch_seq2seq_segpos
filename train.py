from read import Reader
from hyperParams import HyperParams
from optparse import OptionParser
from encoder import Encoder
from decoder import Decoder
from common import  getMaxIndex
from eval import Eval
import torch.nn
import torch.autograd
import torch.nn.functional
import random

class Trainer:
    def __init__(self):
        self.word_state = {}
        self.char_state = {}
        self.bichar_state = {}
        self.pos_state = {}
        self.hyperParams = HyperParams()

    def createAlphabet(self, trainInsts, devInsts, testInsts):
        print("create alpha.................")
        for inst in trainInsts:
            for w in inst.m_words:
                if w not in self.word_state:
                    self.word_state[w] = 1
                else:
                    self.word_state[w] += 1

            for c in inst.m_chars:
                if c not in self.char_state:
                    self.char_state[c] = 1
                else:
                    self.char_state[c] +=1

            for bc in inst.m_bichars:
                if bc not in self.bichar_state:
                    self.bichar_state[bc] = 1
                else:
                    self.bichar_state[bc] += 1

            for l in inst.m_gold:
                self.hyperParams.labelAlpha.from_string(l)

            for pos in inst.m_pos:
                if pos not in self.pos_state:
                    self.pos_state[pos] = 1
                else:
                    self.pos_state[pos] += 1

        #self.addTestAlphabet(devInsts)
        #self.addTestAlphabet(testInsts)

        self.word_state[self.hyperParams.unk] = self.hyperParams.wordCutOff + 1
        self.word_state[self.hyperParams.padding] = self.hyperParams.wordCutOff + 1

        self.char_state[self.hyperParams.unk] = self.hyperParams.charCutOff + 1
        self.char_state[self.hyperParams.padding] = self.hyperParams.charCutOff + 1

        self.bichar_state[self.hyperParams.unk] = self.hyperParams.bicharCutOff + 1
        self.bichar_state[self.hyperParams.padding] = self.hyperParams.bicharCutOff + 1

        self.pos_state[self.hyperParams.unk] = 1
        self.pos_state[self.hyperParams.padding] = 1

        self.hyperParams.wordAlpha.initial(self.word_state, self.hyperParams.wordCutOff)
        self.hyperParams.charAlpha.initial(self.char_state, self.hyperParams.charCutOff)
        self.hyperParams.bicharAlpha.initial(self.bichar_state, self.hyperParams.bicharCutOff)
        self.hyperParams.posAlpha.initial(self.pos_state, 0)


        self.hyperParams.wordUNKID = self.hyperParams.wordAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.charUNKID = self.hyperParams.charAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.posUNKID = self.hyperParams.posAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.bicharUNKID = self.hyperParams.bicharAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.wordPaddingID = self.hyperParams.wordAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.charPaddingID = self.hyperParams.charAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.bicharPaddingID = self.hyperParams.bicharAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.posPaddingID = self.hyperParams.posAlpha.from_string(self.hyperParams.padding)

        self.hyperParams.wordAlpha.set_fixed_flag(True)
        self.hyperParams.charAlpha.set_fixed_flag(True)
        self.hyperParams.bicharAlpha.set_fixed_flag(True)
        self.hyperParams.labelAlpha.set_fixed_flag(True)
        self.hyperParams.posAlpha.set_fixed_flag(True)

        self.hyperParams.wordNum = self.hyperParams.wordAlpha.m_size
        self.hyperParams.charNum = self.hyperParams.charAlpha.m_size
        self.hyperParams.bicharNum = self.hyperParams.bicharAlpha.m_size
        self.hyperParams.labelSize = self.hyperParams.labelAlpha.m_size
        self.hyperParams.posNum = self.hyperParams.posAlpha.m_size

        print("label size: ", self.hyperParams.labelSize)
        print("word size: ", self.hyperParams.wordNum)
        print("char size: ", self.hyperParams.charNum)
        print("bichar size: ", self.hyperParams.bicharNum)
        print("pos size: ", self.hyperParams.posNum)

    def addTestAlphabet(self, testInsts):
        print("add test alpha...")
        for inst in testInsts:
            if not self.hyperParams.charFineTune:
                for c in inst.m_chars:
                    if c not in self.char_state:
                        self.char_state[c] = 1
                    else:
                        self.char_state[c] +=1

            if not self.hyperParams.bicharFineTune:
                for bc in inst.m_bichars:
                    if bc not in self.bichar_state:
                        self.bichar_state[bc] = 1
                    else:
                        self.bichar_state[bc] += 1


    def instance2Example(self, insts):
        for inst in insts:
            for idx in range(inst.m_word_size):
                word = inst.m_words[idx]
                wordID = self.hyperParams.wordAlpha.from_string(word)
                if(wordID == -1):
                    wordID = self.hyperParams.wordUNKID
                inst.m_word_indexes.append(wordID)

                pos = inst.m_pos[idx]
                posID = self.hyperParams.posAlpha.from_string(pos)
                if(wordID == -1):
                    posID = self.hyperParams.posUNKID
                inst.m_pos_indexes.append(posID)

            for idx in range(inst.m_char_size):
                char = inst.m_chars[idx]
                charID = self.hyperParams.charAlpha.from_string(char)
                if(charID == -1):
                    charID = self.hyperParams.charUNKID
                inst.m_char_indexes.append(charID)

            for idx in range(inst.m_bichar_size):
                bichar = inst.m_bichars[idx]
                bicharID = self.hyperParams.bicharAlpha.from_string(bichar)
                if(bicharID == -1):
                    bicharID = self.hyperParams.bicharUNKID
                inst.m_bichar_indexes.append(bicharID)

            for idx in range(inst.m_gold_size):
                gold = inst.m_gold[idx]
                goldID = self.hyperParams.labelAlpha.from_string(gold)
                inst.m_gold_indexes.append(goldID)

    def getBatchFeatLabel(self, insts):
        batch = len(insts)
        max_word_size = -1
        max_char_size = -1
        max_bichar_size = -1
        max_gold_size = -1
        for inst in insts:
            word_size = inst.m_word_size
            if word_size > max_word_size:
                max_word_size = word_size
            char_size = inst.m_char_size
            if char_size > max_char_size:
                max_char_size = char_size
            gold_size = inst.m_gold_size
            if gold_size > max_gold_size:
                max_gold_size = gold_size
            bichar_size = inst.m_bichar_size
            if bichar_size > max_bichar_size:
                max_bichar_size = bichar_size
        batch_word_feats = torch.autograd.Variable(torch.LongTensor(batch, max_word_size))
        batch_pos_feats = torch.autograd.Variable(torch.LongTensor(batch, max_word_size))
        batch_char_feats = torch.autograd.Variable(torch.LongTensor(batch, max_char_size))
        batch_bichar_feats = torch.autograd.Variable(torch.LongTensor(batch, max_bichar_size))
        batch_gold_feats = torch.autograd.Variable(torch.LongTensor(max_gold_size * batch))

        for idx in range(batch):
            inst = insts[idx]
            for idy in range(max_word_size):
                if idy < inst.m_word_size:
                    batch_word_feats.data[idx][idy] = inst.m_word_indexes[idy]
                    batch_pos_feats.data[idx][idy] = inst.m_pos_indexes[idy]
                else:
                    batch_word_feats.data[idx][idy] = self.hyperParams.wordPaddingID
                    batch_pos_feats.data[idx][idy] = self.hyperParams.posPaddingID

            for idy in range(max_char_size):
                if idy < inst.m_char_size:
                    batch_char_feats.data[idx][idy] = inst.m_char_indexes[idy]
                else:
                    batch_char_feats.data[idx][idy] = self.hyperParams.charPaddingID

            for idy in range(max_bichar_size):
                if idy < inst.m_bichar_size:
                    batch_bichar_feats.data[idx][idy] = inst.m_bichar_indexes[idy]
                else:
                    batch_bichar_feats.data[idx][idy] = self.hyperParams.bicharPaddingID

            for idy in range(max_gold_size):
                if idy < inst.m_gold_size:
                    batch_gold_feats.data[idy + idx * max_gold_size] = inst.m_gold_indexes[idy]
                else:
                    batch_gold_feats.data[idy + idx * max_gold_size] = 0
        if self.hyperParams.useCuda:
            return batch_word_feats.cuda(), batch_char_feats.cuda(), batch_bichar_feats.cuda(), batch_gold_feats.cuda(), batch
        else:
            return batch_word_feats, batch_char_feats, batch_bichar_feats, batch_gold_feats, batch

    def train(self, train_file, dev_file, test_file, model_file):
        self.hyperParams.show()
        torch.set_num_threads(self.hyperParams.thread)
        reader = Reader()

        trainInsts = reader.readInstances(train_file, self.hyperParams.maxInstance)
        devInsts = reader.readInstances(dev_file, self.hyperParams.maxInstance)
        testInsts = reader.readInstances(test_file, self.hyperParams.maxInstance)

        print("Training Instance: ", len(trainInsts))
        print("Dev Instance: ", len(devInsts))
        print("Test Instance: ", len(testInsts))

        self.createAlphabet(trainInsts, devInsts, testInsts)

        self.instance2Example(trainInsts)
        self.instance2Example(devInsts)
        self.instance2Example(testInsts)

        self.encoder = Encoder(self.hyperParams)
        self.decoder = Decoder(self.hyperParams)

        if self.hyperParams.useCuda:
            self.encoder.cuda()
            self.decoder.cuda()

        indexes = []
        train_num = len(trainInsts)
        for idx in range(train_num):
            indexes.append(idx)

        encoder_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
        encoder_optimizer = torch.optim.Adam(params=encoder_parameters,
                                             lr=self.hyperParams.learningRate,
                                             weight_decay=self.hyperParams.reg)

        decoder_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        decoder_optimizer = torch.optim.Adam(params=decoder_parameters,
                                             lr=self.hyperParams.learningRate,
                                             weight_decay=self.hyperParams.reg)

        batchBlock = train_num // self.hyperParams.batch
        if train_num % self.hyperParams.batch != 0:
            batchBlock += 1
        train_eval = Eval()
        dev_eval_seg = Eval()
        dev_eval_pos = Eval()
        test_eval_seg = Eval()
        test_eval_pos = Eval()
        for iter in range(self.hyperParams.maxIter):
            print('###Iteration' + str(iter) + '###')
            random.shuffle(indexes)
            self.encoder.train()
            self.decoder.train()
            for updateIter in range(batchBlock):
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                insts = []
                start_pos = updateIter * self.hyperParams.batch
                end_pos = (updateIter + 1) * self.hyperParams.batch
                if end_pos > train_num:
                    end_pos = train_num
                for idx in range(start_pos, end_pos):
                    insts.append(trainInsts[indexes[idx]])
                batch_word_feats, batch_char_feats, batch_bichar_feats, batch_gold, batch = self.getBatchFeatLabel(insts)
                maxCharSize = batch_char_feats.size()[1]
                encoder_hidden = self.encoder.init_hidden(batch)
                encoder_output, encoder_hidden = self.encoder(batch_char_feats, batch_bichar_feats, encoder_hidden, batch)
                decoder_output, _ = self.decoder(insts, encoder_output, batch, True)
                train_eval.clear()
                for idx in range(batch):
                    inst = insts[idx]
                    for idy in range(inst.m_char_size):
                        actionID = getMaxIndex(self.hyperParams, decoder_output[idx * maxCharSize + idy])
                        if actionID == inst.m_gold_indexes[idy]:
                            train_eval.correct_num += 1
                    train_eval.gold_num += inst.m_char_size
                loss = torch.nn.functional.nll_loss(decoder_output, batch_gold)
                print("current: ", updateIter + 1, "cost: ", loss.data[0], "correct: ", train_eval.acc())
                loss.backward()

                torch.nn.utils.clip_grad_norm(encoder_parameters, self.hyperParams.clip)
                torch.nn.utils.clip_grad_norm(decoder_parameters, self.hyperParams.clip)

                encoder_optimizer.step()
                decoder_optimizer.step()

            self.encoder.eval()
            self.decoder.eval()
            dev_eval_seg.clear()
            dev_eval_pos.clear()
            for idx in range(len(devInsts)):
                inst = devInsts[idx]
                state = self.predict(inst)
                devInsts[idx].jointPRF(state[0], dev_eval_seg, dev_eval_pos)
            p, r, f = dev_eval_seg.getFscore()
            print('seg dev: precision = ', str(p), ', recall = ', str(r), ', f-score = ', str(f))
            p, r, f = dev_eval_pos.getFscore()
            print('pos dev: precision = ', str(p), ', recall = ', str(r), ', f-score = ', str(f))

            test_eval_seg.clear()
            test_eval_pos.clear()
            for idx in range(len(testInsts)):
                inst = testInsts[idx]
                state = self.predict(inst)
                testInsts[idx].jointPRF(state[0], test_eval_seg, test_eval_pos)
            p, r, f = test_eval_seg.getFscore()
            print('seg test: precision = ', str(p), ', recall = ', str(r), ', f-score = ', str(f))
            p, r, f = test_eval_pos.getFscore()
            print('pos test: precision = ', str(p), ', recall = ', str(r), ', f-score = ', str(f))



    def predict(self, inst):
        insts = []
        insts.append(inst)
        batch_word_feats, batch_char_feats, batch_bichar_feats, batch_gold, batch = self.getBatchFeatLabel(insts)
        encoder_hidden = self.encoder.init_hidden(batch)
        encoder_output, encoder_hidden = self.encoder(batch_char_feats, batch_bichar_feats, encoder_hidden, batch)
        decoder_output, state = self.decoder(insts, encoder_output, batch, False)
        return state

parser = OptionParser()
parser.add_option("--train", dest="trainFile",
                  help="train dataset")

parser.add_option("--dev", dest="devFile",
                  help="dev dataset")

parser.add_option("--test", dest="testFile",
                  help="test dataset")

parser.add_option("--model", dest="modelFile",
                  help="model file")
parser.add_option(
    "-l", "--learn", dest="learn", help="learn or test", action="store_false", default=True)

random.seed(0)
torch.manual_seed(0)
(options, args) = parser.parse_args()
l = Trainer()
if options.learn:
    l.train(options.trainFile, options.devFile, options.testFile, options.modelFile)
else:
    l.test(options.testFile,options.modelFile)

