
class Instance:
    def __init__(self):
        self.m_chars = []
        self.m_char_size = 0

        self.m_char_types = []
        self.m_char_type_size = 0

        self.m_bichars = []
        self.m_bichar_size = 0
        self.m_words = []
        self.m_word_size = 0

        self.m_gold = []
        self.m_pos = []
        self.m_gold_seg = []
        self.m_gold_pos = []

        self.m_char_indexes = []
        self.m_extchar_indexes = []
        self.m_char_type_indexes = []

        self.m_bichar_indexes = []
        self.m_extbichar_indexes = []

        self.m_word_indexes = []
        self.m_pos_indexes = []

        self.m_gold_indexes = []
        self.m_gold_size = 0

    def show(self):
        print(self.m_chars)
        print(self.m_bichars)
        print(self.m_words)
        print(self.m_gold)

    def evalPRF(self, state, seg_eval):
        words = state.words
        count = 0
        predict_seg = []
        for w in words:
            predict_seg.append('[' + str(count) + ',' + str(count + len(w)) + ']')
            count += len(w)
        seg_eval.gold_num += len(self.m_gold_seg)
        seg_eval.predict_num += len(predict_seg)
        for p in predict_seg:
            if p in self.m_gold_seg:
                seg_eval.correct_num += 1

    def jointPRF(self, state, seg_eval, pos_eval):
        words = state.words
        posLabels = state.pos_labels
        count = 0
        predict_seg = []
        predict_pos = []

        for idx in range(len(words)):
            w = words[idx]
            posLabel = posLabels[idx]
            predict_seg.append('[' + str(count) + ',' + str(count + len(w)) + ']')
            predict_pos.append('[' + str(count) + ',' + str(count + len(w)) + ']' + posLabel)
            count += len(w)
        seg_eval.gold_num += len(self.m_gold_seg)
        seg_eval.predict_num += len(predict_seg)
        for p in predict_seg:
            if p in self.m_gold_seg:
                seg_eval.correct_num += 1

        pos_eval.gold_num += len(self.m_gold_pos)
        pos_eval.predict_num += len(predict_pos)
        for p in predict_pos:
            if p in self.m_gold_pos:
                pos_eval.correct_num += 1

class BatchFeats:
    def __init__(self):
        self.batch = 0
        self.pos_feats = 0
        self.char_feats = 0
        self.extchar_feats = 0
        self.char_type_feats = 0
        self.bichar_feats = 0
        self.extbichar_feats = 0
        self.gold_feats = 0

    def cuda(self,gpuID):
        self.pos_feats = self.pos_feats.cuda(gpuID)
        self.char_feats = self.char_feats.cuda(gpuID)
        self.extchar_feats = self.extchar_feats.cuda(gpuID)
        self.char_type_feats = self.char_type_feats.cuda(gpuID)
        self.bichar_feats = self.bichar_feats.cuda(gpuID)
        self.extbichar_feats = self.extbichar_feats.cuda(gpuID)
        self.gold_feats = self.gold_feats.cuda(gpuID)

# class Example:
#     def __init__(self):
#         self.m_char_indexes = []
#         self.m_char_size = 0
#
#         self.m_bichar_indexes = []
#         self.m_bichar_size = 0
#
#         self.m_word_indexes = []
#         self.m_pos_indexes = []
#         self.m_word_size = 0
#
#         self.m_gold = []
#         self.m_gold_size = 0
#
#     def show(self):
#         print(self.m_char_indexes)
#         print(self.m_char_size)
#         print(self.m_bichar_indexes)
#         print(self.m_bichar_size)
#         print(self.m_word_indexes)
#         print(self.m_word_size)

