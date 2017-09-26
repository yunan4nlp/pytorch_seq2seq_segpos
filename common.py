import string

def getMaxIndex(hyperParams, decoder_output):
    max = decoder_output.data[0]
    maxIndex = 0
    for idx in range(1, hyperParams.labelSize):
        if decoder_output.data[idx] > max:
            max = decoder_output.data[idx]
            maxIndex = idx
    return maxIndex

def wordtype(str):
    str_type = ''
    for char in str:
        charLen = len(char.encode('utf8'))
        if charLen > 2:
            str_type += 'U'
        elif charLen == 2:
            str_type += 'u'
        elif char.isalpha():
            if char.isupper():
                str_type += 'E'
            else:
                str_type += 'e'
        elif char.isdigit():
            str_type += 'd'
        elif char in string.punctuation:
            str_type += 'p'
        else:
            str_type += 'o'
    return str_type
