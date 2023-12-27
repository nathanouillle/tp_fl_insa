def attack(predictions,threshold):
    '''
    Submission is a list composed of 1 (train) and 0 (test), with the same len as predictions (number of data in the
    dataset). It's contains the result of the MIA.
'''

    import numpy as np
    # If the prediction is above the treshold, we consider it as a train data
    # If the prediction is below the treshold, we consider it as a test data

    max = np.max(predictions, axis=1)
    train_indices = np.where(max >= threshold)[0]

    submission = np.zeros(len(predictions), dtype=int)
    submission[train_indices] = 1

    return submission