class HyperParams:
    def __init__(self):
        self.wordNum = 0
        self.charNum = 0
        self.bicharNum = 0
        self.posNum = 0
        self.labelSize = 0

        self.unk = '-unk-'
        self.padding = '-padding-'

        self.start = '-start-'
        self.end = '-end-'
        self.maxLen = 20



        self.clip = 10
        self.maxIter = 1000
        self.verboseIter = 20

        self.wordCutOff = 0
        self.wordEmbSize = 50
        self.wordFineTune = True
        self.wordEmbFile = ""
        self.wordUNKID = 0
        self.wordPaddingID = 0

        self.bicharCutOff = 0
        self.bicharEmbSize = 200
        self.bicharFineTune = False
        self.bicharEmbFile = "E:\\py_workspace\\Seq2Seq_len\\data\\emb\\bichar.sample"
        #self.bicharEmbFile = "E:\\py_workspace\\Seq2Seq_bmes\\data\\bichar.vec"
        self.bicharUNKID = 0
        self.bicharPaddingID = 0

        self.charCutOff = 0
        self.charEmbSize = 200
        self.charFineTune = False
        #self.charEmbFile = ""
        self.charUNKID = 0
        self.charPaddingID = 0
        self.charEmbFile = "E:\\py_workspace\\Seq2Seq_len\\data\\emb\\char.vec"
        #self.charEmbFile = "E:\\py_workspace\\Seq2Seq_bmes\\data\\char.vec"

        self.posEmbSize = 100
        self.posFineTune = True

        self.dropProb = 0.25
        self.rnnHiddenSize = 200
        self.hiddenSize = 200
        self.thread = 1
        self.learningRate = 0.001
        self.reg = 1e-8
        self.maxInstance = 2
        self.batch = 2
        self.useCuda = False

        self.wordAlpha = Alphabet()
        self.charAlpha = Alphabet()
        self.bicharAlpha = Alphabet()

        self.labelAlpha = Alphabet()
        self.posAlpha = Alphabet()

    def show(self):
        print('wordCutOff = ', self.wordCutOff)
        print('wordEmbSize = ', self.wordEmbSize)
        print('wordFineTune = ', self.wordFineTune)
        print('wordFile = ', self.wordEmbFile)

        print('charCutOff = ', self.charCutOff)
        print('charEmbSize = ', self.charEmbSize)
        print('charFineTune = ', self.charFineTune)
        print('charFile = ', self.charEmbFile)

        print('bicharCutOff = ', self.bicharCutOff)
        print('bicharEmbSize = ', self.bicharEmbSize)
        print('bicharFineTune = ', self.bicharFineTune)
        print('bicharFile = ', self.bicharEmbFile)

        print('maxLen', self.maxLen)
        print('rnnHiddenSize = ', self.rnnHiddenSize)
        print('learningRate = ', self.learningRate)
        print('batch = ', self.batch)

        print('maxInstance = ', self.maxInstance)
        print('maxIter =', self.maxIter)
        print('thread = ', self.thread)
        print('verboseIter = ', self.verboseIter)


class Alphabet:
    def __init__(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = {}

    def from_id(self, qid, defineStr = ''):
        if int(qid) < 0 or self.m_size <= qid:
            return defineStr
        else:
            return self.id2string[qid]

    def from_string(self, str):
        if str in self.string2id:
            return self.string2id[str]
        else:
            if not self.m_b_fixed:
                newid = self.m_size
                self.id2string.append(str)
                self.string2id[str] = newid
                self.m_size += 1
                if self.m_size >= self.max_cap:
                    self.m_b_fixed = True
                return newid
            else:
                return -1

    def clear(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = {}

    def set_fixed_flag(self, bfixed):
        self.m_b_fixed = bfixed
        if (not self.m_b_fixed) and (self.m_size >= self.max_cap):
            self.m_b_fixed = True

    def initial(self, elem_state, cutoff = 0):
        for key in elem_state:
            if  elem_state[key] > cutoff:
                self.from_string(key)
        self.set_fixed_flag(True)

    def write(self, path):
        outf = open(path, encoding='utf-8', mode='w')
        for idx in range(self.m_size):
            outf.write(self.id2string[idx] + " " + str(idx) + "\n")
        outf.close()

    def read(self, path):
        inf = open(path, encoding='utf-8', mode='r')
        for line in inf.readlines():
            info = line.split(" ")
            self.id2string.append(info[0])
            self.string2id[info[0]] = int(info[1])
        inf.close()
        self.set_fixed_flag(True)
        self.m_size = len(self.id2string)
