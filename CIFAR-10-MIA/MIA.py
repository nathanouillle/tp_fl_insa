def attack(predictions,treshold):
    '''
    Submission is a list composed of 1 (train) and 0 (test), with the same len as predictions (number of data in the
    dataset). It's contains the result of the MIA.

    '''

    submission = [0]*len(predictions)
    return submission