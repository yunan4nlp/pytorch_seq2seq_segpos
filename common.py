def getMaxIndex(hyperParams, decoder_output):
    max = decoder_output.data[0]
    maxIndex = 0
    for idx in range(1, hyperParams.labelSize):
        if decoder_output.data[idx] > max:
            max = decoder_output.data[idx]
            maxIndex = idx
    return maxIndex