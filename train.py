from read import Reader
from hyperParams import HyperParams
from instance import BatchFeats
from optparse import OptionParser
from encoder2 import Encoder
from decoder2 import Decoder
from common import  getMaxIndex
from common import  paddingkey
from common import  unkkey
from common import  app
from eval import Eval
import torch.nn
import torch.autograd
import torch.nn.functional
import random
import time

class Trainer:
    def __init__(self):
        self.word_state = {}
        self.char_state = {}
        self.bichar_state = {}
        self.chartype_state = {}
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

            for bc in inst.m_left_bichars:
                if bc not in self.bichar_state:
                    self.bichar_state[bc] = 1
                else:
                    self.bichar_state[bc] += 1
            bc = inst.m_right_bichars[-1]
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

        self.chartype_state['U'] = 1
        self.chartype_state['u'] = 1
        self.chartype_state['E'] = 1
        self.chartype_state['e'] = 1
        self.chartype_state['p'] = 1
        self.chartype_state['d'] = 1
        self.chartype_state['o'] = 1
        self.chartype_state[paddingkey] = 1

        self.hyperParams.charTypeAlpha.initial(self.chartype_state)

        self.word_state[unkkey] = self.hyperParams.wordCutOff + 1
        self.word_state[paddingkey] = self.hyperParams.wordCutOff + 1

        self.char_state[unkkey] = self.hyperParams.charCutOff + 1
        self.char_state[paddingkey] = self.hyperParams.charCutOff + 1

        self.bichar_state[unkkey] = self.hyperParams.bicharCutOff + 1
        self.bichar_state[paddingkey] = self.hyperParams.bicharCutOff + 1

        self.pos_state[unkkey] = 1
        self.pos_state[paddingkey] = 1

        self.hyperParams.wordAlpha.initial(self.word_state, self.hyperParams.wordCutOff)
        self.hyperParams.charAlpha.initial(self.char_state, self.hyperParams.charCutOff)
        self.hyperParams.bicharAlpha.initial(self.bichar_state, self.hyperParams.bicharCutOff)
        self.hyperParams.posAlpha.initial(self.pos_state, 0)

        self.hyperParams.extCharAlpha.initial_from_pretrain(self.hyperParams.charEmbFile,
                                                            unkkey,
                                                            paddingkey)
        self.hyperParams.extBicharAlpha.initial_from_pretrain(self.hyperParams.bicharEmbFile,
                                                              unkkey,
                                                              paddingkey)


        self.hyperParams.wordUNKID = self.hyperParams.wordAlpha.from_string(unkkey)
        self.hyperParams.charUNKID = self.hyperParams.charAlpha.from_string(unkkey)
        self.hyperParams.extCharUNKID = self.hyperParams.extCharAlpha.from_string(unkkey)
        self.hyperParams.posUNKID = self.hyperParams.posAlpha.from_string(unkkey)
        self.hyperParams.bicharUNKID = self.hyperParams.bicharAlpha.from_string(unkkey)
        self.hyperParams.extBicharUNKID = self.hyperParams.extBicharAlpha.from_string(unkkey)

        self.hyperParams.wordPaddingID = self.hyperParams.wordAlpha.from_string(paddingkey)
        self.hyperParams.charPaddingID = self.hyperParams.charAlpha.from_string(paddingkey)
        self.hyperParams.extCharPaddingID = self.hyperParams.extCharAlpha.from_string(paddingkey)
        self.hyperParams.bicharPaddingID = self.hyperParams.bicharAlpha.from_string(paddingkey)
        self.hyperParams.extBicharPaddingID = self.hyperParams.extBicharAlpha.from_string(paddingkey)

        self.hyperParams.posPaddingID = self.hyperParams.posAlpha.from_string(paddingkey)
        self.hyperParams.charTypePaddingID = self.hyperParams.charTypeAlpha.from_string(paddingkey)
        self.hyperParams.appID = self.hyperParams.labelAlpha.from_string(app)

        self.hyperParams.wordAlpha.set_fixed_flag(True)
        self.hyperParams.charAlpha.set_fixed_flag(True)
        self.hyperParams.bicharAlpha.set_fixed_flag(True)
        self.hyperParams.extCharAlpha.set_fixed_flag(True)
        self.hyperParams.extBicharAlpha.set_fixed_flag(True)

        self.hyperParams.labelAlpha.set_fixed_flag(True)
        self.hyperParams.posAlpha.set_fixed_flag(True)
        self.hyperParams.charTypeAlpha.set_fixed_flag(True)

        self.hyperParams.wordNum = self.hyperParams.wordAlpha.m_size
        self.hyperParams.charNum = self.hyperParams.charAlpha.m_size
        self.hyperParams.bicharNum = self.hyperParams.bicharAlpha.m_size
        self.hyperParams.extCharNum = self.hyperParams.extCharAlpha.m_size
        self.hyperParams.extBicharNum = self.hyperParams.extBicharAlpha.m_size

        self.hyperParams.labelSize = self.hyperParams.labelAlpha.m_size
        self.hyperParams.posNum = self.hyperParams.posAlpha.m_size
        self.hyperParams.charTypeNum = self.hyperParams.charTypeAlpha.m_size


        print("label size: ", self.hyperParams.labelSize)
        print("word size: ", self.hyperParams.wordNum)
        print("char size: ", self.hyperParams.charNum)
        print("ext char size: ", self.hyperParams.extCharNum)
        print("bichar size: ", self.hyperParams.bicharNum)
        print("ext bichar size: ", self.hyperParams.extBicharNum)
        print("pos size: ", self.hyperParams.posNum)
        print("char type size: ", self.hyperParams.charTypeNum)
        print("app ID: ", self.hyperParams.appID)

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
                extCharID = self.hyperParams.extCharAlpha.from_string(char)
                if(charID == -1):
                    charID = self.hyperParams.charUNKID
                if(extCharID == -1):
                    extCharID = self.hyperParams.extCharUNKID
                inst.m_char_indexes.append(charID)
                inst.m_extchar_indexes.append(extCharID)

            for idx in range(inst.m_bichar_size):
                leftbichar = inst.m_left_bichars[idx]
                leftbicharID = self.hyperParams.bicharAlpha.from_string(leftbichar)
                leftextBicharID = self.hyperParams.extBicharAlpha.from_string(leftbichar)
                if(leftbicharID == -1):
                    leftbicharID = self.hyperParams.bicharUNKID
                if(leftextBicharID == -1):
                    leftextBicharID = self.hyperParams.extBicharUNKID
                inst.m_leftbichar_indexes.append(leftbicharID)
                inst.m_leftextbichar_indexes.append(leftextBicharID)

            for idx in range(inst.m_bichar_size):
                rightbichar = inst.m_right_bichars[idx]
                rightbicharID = self.hyperParams.bicharAlpha.from_string(rightbichar)
                rightextBicharID = self.hyperParams.extBicharAlpha.from_string(rightbichar)
                if(rightbicharID == -1):
                    rightbicharID = self.hyperParams.bicharUNKID
                if(rightextBicharID == -1):
                    rightextBicharID = self.hyperParams.extBicharUNKID
                inst.m_rightbichar_indexes.append(rightbicharID)
                inst.m_rightextbichar_indexes.append(rightextBicharID)

            for idx in range(inst.m_char_type_size):
                charType = inst.m_char_types[idx]
                charTypeID = self.hyperParams.charTypeAlpha.from_string(charType)
                inst.m_char_type_indexes.append(charTypeID)


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
        batch_pos_feats = torch.autograd.Variable(torch.LongTensor(batch, max_word_size))
        batch_char_feats = torch.autograd.Variable(torch.LongTensor(batch, max_char_size))
        batch_extchar_feats = torch.autograd.Variable(torch.LongTensor(batch, max_char_size))
        batch_char_type_feats = torch.autograd.Variable(torch.LongTensor(batch, max_char_size))
        batch_leftbichar_feats = torch.autograd.Variable(torch.LongTensor(batch, max_bichar_size))
        batch_leftextbichar_feats = torch.autograd.Variable(torch.LongTensor(batch, max_bichar_size))
        batch_rightbichar_feats = torch.autograd.Variable(torch.LongTensor(batch, max_bichar_size))
        batch_rightextbichar_feats = torch.autograd.Variable(torch.LongTensor(batch, max_bichar_size))
        batch_gold_feats = torch.autograd.Variable(torch.LongTensor(max_gold_size * batch))

        for idx in range(batch):
            inst = insts[idx]
            for idy in range(max_word_size):
                if idy < inst.m_word_size:
                    batch_pos_feats.data[idx][idy] = inst.m_pos_indexes[idy]
                else:
                    batch_pos_feats.data[idx][idy] = self.hyperParams.posPaddingID

            for idy in range(max_char_size):
                if idy < inst.m_char_size:
                    batch_char_feats.data[idx][idy] = inst.m_char_indexes[idy]
                    batch_extchar_feats.data[idx][idy] = inst.m_extchar_indexes[idy]
                else:
                    batch_char_feats.data[idx][idy] = self.hyperParams.charPaddingID
                    batch_extchar_feats.data[idx][idy] = self.hyperParams.extCharPaddingID

            for idy in range(max_char_size):
                if idy < inst.m_char_size:
                    batch_char_type_feats.data[idx][idy] = inst.m_char_type_indexes[idy]
                else:
                    batch_char_type_feats.data[idx][idy] = self.hyperParams.charTypePaddingID

            for idy in range(max_bichar_size):
                if idy < inst.m_bichar_size:
                    batch_leftbichar_feats.data[idx][idy] = inst.m_leftbichar_indexes[idy]
                    batch_leftextbichar_feats.data[idx][idy] = inst.m_leftextbichar_indexes[idy]
                else:
                    batch_leftbichar_feats.data[idx][idy] = self.hyperParams.bicharPaddingID
                    batch_leftextbichar_feats.data[idx][idy] = self.hyperParams.extBicharPaddingID

            for idy in range(max_bichar_size):
                if idy < inst.m_bichar_size:
                    batch_rightbichar_feats.data[idx][idy] = inst.m_rightbichar_indexes[idy]
                    batch_rightextbichar_feats.data[idx][idy] = inst.m_rightextbichar_indexes[idy]
                else:
                    batch_rightbichar_feats.data[idx][idy] = self.hyperParams.bicharPaddingID
                    batch_rightextbichar_feats.data[idx][idy] = self.hyperParams.extBicharPaddingID

            for idy in range(max_gold_size):
                if idy < inst.m_gold_size:
                    batch_gold_feats.data[idy + idx * max_gold_size] = inst.m_gold_indexes[idy]
                else:
                    batch_gold_feats.data[idy + idx * max_gold_size] = 0

        feats = BatchFeats()
        feats.batch = batch
        feats.char_feats = batch_char_feats
        feats.extchar_feats = batch_extchar_feats

        feats.leftbichar_feats = batch_leftbichar_feats
        feats.rightbichar_feats = batch_rightbichar_feats
        feats.leftextbichar_feats = batch_leftextbichar_feats
        feats.rightextbichar_feats = batch_rightextbichar_feats

        feats.pos_feats = batch_pos_feats
        feats.char_type_feats = batch_char_type_feats
        feats.gold_feats = batch_gold_feats

        if self.hyperParams.useCuda:
            feats.cuda(self.hyperParams.gpuID)
        return feats
        # if self.hyperParams.useCuda:
        #     return batch_char_type_feats.cuda(self.hyperParams.gpuID), batch_char_feats.cuda(self.hyperParams.gpuID), batch_bichar_feats.cuda(self.hyperParams.gpuID), batch_gold_feats.cuda(self.hyperParams.gpuID), batch
        # else:
        #     return batch_char_type_feats, batch_char_feats, batch_bichar_feats, batch_gold_feats, batch

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
            self.encoder.cuda(self.hyperParams.gpuID)
            self.decoder.cuda(self.hyperParams.gpuID)

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
            start = time.time()
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
                feats = self.getBatchFeatLabel(insts)
                #print(batch_gold)
                maxCharSize = feats.char_feats.size()[1]
                #encoder_hidden = self.encoder.init_hidden(feats.batch)
                encoder_output = self.encoder(feats)
                decoder_output, _ = self.decoder(insts, encoder_output, feats.batch, True)
                train_eval.clear()
                for idx in range(feats.batch):
                    inst = insts[idx]
                    for idy in range(inst.m_char_size):
                        actionID = getMaxIndex(self.hyperParams, decoder_output[idx * maxCharSize + idy])
                        if actionID == inst.m_gold_indexes[idy]:
                            train_eval.correct_num += 1
                    train_eval.gold_num += inst.m_char_size
                loss = torch.nn.functional.nll_loss(decoder_output, feats.gold_feats)
                print("current: ", updateIter + 1, "cost: ", loss.data[0], "correct: ", train_eval.acc())
                loss.backward()

                torch.nn.utils.clip_grad_norm(encoder_parameters, self.hyperParams.clip)
                torch.nn.utils.clip_grad_norm(decoder_parameters, self.hyperParams.clip)

                encoder_optimizer.step()
                decoder_optimizer.step()
            print("train time cost: ", time.time() - start, 's')

            self.encoder.eval()
            self.decoder.eval()
            start = time.time()
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
            print('dev cost time', time.time() - start, 's')

            test_eval_seg.clear()
            test_eval_pos.clear()
            start = time.time()
            for idx in range(len(testInsts)):
                inst = testInsts[idx]
                state = self.predict(inst)
                testInsts[idx].jointPRF(state[0], test_eval_seg, test_eval_pos)
            p, r, f = test_eval_seg.getFscore()
            print('seg test: precision = ', str(p), ', recall = ', str(r), ', f-score = ', str(f))
            p, r, f = test_eval_pos.getFscore()
            print('pos test: precision = ', str(p), ', recall = ', str(r), ', f-score = ', str(f))
            print('test cost time', time.time() - start, 's')



    def predict(self, inst):
        insts = []
        insts.append(inst)
        feats = self.getBatchFeatLabel(insts)
        encoder_output = self.encoder(feats)
        decoder_output, state = self.decoder(insts, encoder_output, feats.batch, False)
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

