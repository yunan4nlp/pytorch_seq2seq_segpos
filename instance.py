
class Instance:
    def __init__(self):
        self.m_chars = []
        self.m_char_size = 0
        self.m_bichars = []
        self.m_bichar_size = 0
        self.m_words = []
        self.m_word_size = 0

        self.m_gold = []
        self.m_pos = []
        self.m_gold_str = []

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

    def evalPRF(self, words, eval):
        count = 0
        predict_str = []
        for w in words:
            predict_str.append('[' + str(count) + ',' + str(count + len(w)) + ']')
            count += len(w)
        eval.gold_num += len(self.m_gold_str)
        eval.predict_num += len(predict_str)
        for p in predict_str:
            if p in self.m_gold_str:
                eval.correct_num += 1

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

