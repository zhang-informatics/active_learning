"""
The ActiveLearningModel class. This class acts as an interface to 
a machine-learning classifier instance (e.g. from scikit-learn), a
QueryStrategy instance, and a Data instance.
"""

import numpy as np
from sklearn import base, metrics, model_selection

from .containers import Data
from .querystrategies import QueryStrategy, SimpleMargin


class ActiveLearningModel(object):

    def __init__(self, classifier, query_strategy, U_proportion=0.9, random_state=None):
        '''
        :param sklearn.base.BaseEstimator classifier: Classifier to build the model.
        :param QueryStrategy query_strategy: QueryStrategy instance to use.
        :param float U_proportion: proportion of training data to be assigned
                                   the unlabeled set.
        :param int random_state: Sets the random_state parameter of train_test_split.
        '''
        self.__check_args(classifier, query_strategy, U_proportion)
        self.classifier = base.clone(classifier)
        self.query_strategy = query_strategy
        self.U_proportion = U_proportion
        self.random_state = random_state
        self.L = Data()  # Labeled data.
        self.U = Data()  # Unlabeled data.
        self.T = Data()  # Test data.
        self.classes = None

    def __check_args(self, classifier, query_strategy, U_proportion):
        if not isinstance(query_strategy, QueryStrategy):
            raise ValueError("query_strategy must be an instance of QueryStrategy.")
        if not 0 < U_proportion < 1:
            raise ValueError("U_proportion must be in range (0,1) exclusive. Got {}."
                              .format(U_proportion))
        if isinstance(query_strategy, SimpleMargin) and \
          not hasattr(classifier, "decision_function"):
            raise ValueError("{} compatible only with discriminative models."
                              .format(str(query_strategy)))

    def split_data(self, train_x, test_x, train_y, test_y):
        '''
        Splits data into unlabeled, labeled, and test sets
        according to self.U_proportion.

        :param np.array train_x: Training data features.
        :param np.array test_x: Test data features.
        :param np.array train_y: Training data labels.
        :param np.array test_y: Test data labels.
        '''
        U_size = int(np.ceil(self.U_proportion * train_x.shape[0]))
        if not 0 < U_size < train_x.shape[0]:
            raise ValueError("U_proportion must result in non-empty labeled and unlabeled sets.")
        if train_x.shape[0] - U_size <= 1:
            raise ValueError("U_proportion must result in a labeled set with > 1 members.")
        temp = model_selection.train_test_split(train_x, train_y,
                                                test_size=U_size,
                                                random_state=self.random_state)
        self.L.x, self.U.x, self.L.y, self.U.y = temp
        self.T.x = test_x
        self.T.y = test_y

    def update_labels(self):
        '''
        Gets the chosen index from the query strategy,
        adds the corresponding data point to L and removes
        it from U. Logs which instance is picked from U.

        :returns: chosen x and y, for use with partial_train()
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        '''
        index = self.query_strategy.query(self.U, self.L, self.classifier)
        chosen_x = self.U.x[index]
        chosen_y = np.array([self.U.y[index]])
        self.L.y = np.append(self.L.y, chosen_y, axis=0)
        self.L.x = np.vstack((self.L.x, chosen_x))
        self.U.x = np.delete(self.U.x, index, axis=0)
        self.U.y = np.delete(self.U.y, index, axis=0)
        return chosen_x.reshape(1, -1), chosen_y
        
    def train(self):
        '''
        Trains the classifier on L.
        '''
        self.classifier.fit(self.L.x, self.L.y)

    def partial_train(self, new_x, new_y):
        '''
        Given a subset of training examples, calls partial_fit.
        
        :param numpy.ndarray new_x: Feature array.
        :param numpy.ndarray new_y: Label array.
        '''
        if self.classes is None:
            self.classes = np.unique(self.U.y)
        self.classifier.partial_fit(new_x, new_y, classes=self.classes)

    def score(self):
        '''
        Computes Area Under the ROC Curve for the current classifier.
        :returns: AUC score.
        :rtype: float
        '''
        try:  # If the classifier is probabilistic.
            scores = self.classifier.predict_proba(self.T.x)[:, 1]
        except AttributeError:
            scores = self.classifier.decision_function(self.T.x)
        auc = metrics.roc_auc_score(self.T.y, scores)
        return auc

    def _get_choice_order(self, ndraws):
        mask = np.ones(self.L.y.shape, dtype=bool)
        L_0_index = self.L.y.shape[0] - ndraws
        mask[:L_0_index] = False
        choice_order = {'x': self.L.x[mask], 'y': self.L.y[mask]}
        return choice_order

    def run(self, train_x, test_x, train_y, test_y, ndraws=None):
        '''
        Run the active learning model. Saves AUC scores for
        each sampling iteration.

        :param np.array train_x: Training data features.
        :param np.array test_x: Test data features.
        :param np.array train_y: Training data labels.
        :param np.array test_y: Test data labels.
        :param int ndraws: Number of times to query the unlabeled set.
                            If None, query entire unlabeled set.
        :returns: AUC scores for each sampling iteration.
        :rtype: numpy.ndarray(shape=(ndraws, ))
        '''
        # Populate L, U, and T
        self.split_data(train_x, test_x, train_y, test_y)
        if ndraws is None:
            ndraws = self.U.x.shape[0]
        scores = np.zeros(ndraws, dtype=np.float32)
        for i in range(ndraws):
            self.train()
            auc = self.score()
            scores[i] = auc
            x, y = self.update_labels()
        choice_order = self._get_choice_order(ndraws)
        return scores, choice_order
