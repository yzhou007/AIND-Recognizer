import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # TODO implement model selection based on BIC scores
        best_bic = float("+inf")
        best_num_components = self.min_n_components
        for num_state in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state).fit(self.X, self.lengths)
                num_parameters = num_state ** 2 + 2 * len(self.X[0]) * num_state 
                try:
                    logL = model.score(self.X, self.lengths)
                    bic = -2 * model.score(self.X, self.lengths) + num_parameters * math.log10(len(self.X))
                    if best_bic > bic: # get minimum BIC score
                        best_bic = bic
                        best_num_components = num_state
                except:
                    continue
            except:
                continue
        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #warnings.filterwarnings("ignore", category=RuntimeWarning)
        # TODO implement model selection based on DIC scores
        best_dic = float("-inf")
        best_num_components = self.min_n_components
        for num_state in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state).fit(self.X, self.lengths)
                try:
                    logL_Xi = model.score(self.X, self.lengths)
                    logL_Sum = 0.0
                    for key in self.hwords:
                        if key != self.this_word:
                            X_j, lengths_j = self.hwords[key]
                            logL_Sum += model.score(X_j, lengths_j)
                    dic = logL_Xi - logL_Sum / (len(self.hwords) - 1)
                    if best_dic < dic: # get maximum DIC score
                        best_dic = dic
                        best_num_components = num_state
                except:
                    continue
            except:
                continue
        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #warnings.filterwarnings("ignore", category=RuntimeWarning)
        # TODO implement model selection using CV
        if len(self.sequences) < 3: # if the size of the data sequence of a word is less than 3
            best_num_components = self.n_constant
            return self.base_model(best_num_components)
        split_method = KFold()
        best_logL = float("-inf")
        best_num_components = self.min_n_components
        for num_state in range(self.min_n_components, self.max_n_components + 1):
            logL_KFold = [];
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_X, train_Xlengths = combine_sequences(cv_train_idx, self.sequences)
                test_X, test_Xlengths = combine_sequences(cv_test_idx, self.sequences)
                try:
                    model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state).fit(train_X, train_Xlengths)
                    logL_KFold.append(model.score(test_X, test_Xlengths))
                except:
                    continue
            if not logL_KFold: # if there is no score
                continue
            if best_logL < np.mean(logL_KFold): # get maximum cross validation score
                best_logL = np.mean(logL_KFold)
                best_num_components = num_state
        return self.base_model(best_num_components)