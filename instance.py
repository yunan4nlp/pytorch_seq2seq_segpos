
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

        self.m_bichar_indexes = []

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

