import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    test_data = test_set._hmm_data
    for key in test_data:
        test_X, test_Xlengths = test_data[key]
        d = {}
        best_logL = float("-inf")
        best_guess = ""
        for key_model in models:
            try:
                logL = models[key_model].score(test_X, test_Xlengths)
                d[key_model] = logL
                if best_logL < logL:
                    best_logL = logL
                    best_guess = key_model
            except:
                d[key_model] = float("-inf")
        probabilities.append(d)
        guesses.append(best_guess)
            
    return probabilities,guesses
