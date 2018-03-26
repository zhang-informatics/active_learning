import unittest
import numpy as np
from sklearn.svm import SVC

from activelearning.activelearning import ActiveLearningModel
from activelearning.querystrategies import *


class ActiveLearningModelTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(311)
        self.clf = SVC(probability=True)
        self.kwargs = {'U_proportion': 0.4}
        self.train_x = np.array([[0.5, 1.5], [0.5, 3], [2, 3],
                                 [2, 1.5], [3, 1], [3, 3]])
        self.train_y = np.array([0, 0, 0, 1, 1, 1])
        self.test_x = np.array([[1, 2], [2, 1]])
        self.test_y = np.array([0, 1])
        self.ndraws = int(np.ceil(0.4 * self.train_x.shape[0]))

    def tearDown(self):
        del self.clf

    def test_random(self):
        qs = Random()
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_minmax(self):
        qs_kwargs = {'metric': 'mahalanobis'}
        qs = MinMax(**qs_kwargs) 
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(al.query_strategy.distance_metric, 'mahalanobis')

    def test_simple_margin(self):
        qs = SimpleMargin()
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_margin(self):
        qs = Margin()
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_entropy(self):
        qs_kwargs = {'model_change': False}
        qs = Entropy(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_lc(self):
        qs_kwargs = {'model_change': False}
        qs = LeastConfidence(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_lcb(self):
        qs_kwargs = {'model_change': False}
        qs = LeastConfidenceBias(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_lcb2(self):
        qs_kwargs = {'model_change': False}
        qs = LeastConfidenceDynamicBias(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_entropy_mc(self):
        qs_kwargs = {'model_change': True}
        qs = Entropy(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.model_change, True)

    def test_lc_mc(self):
        qs_kwargs = {'model_change': True}
        qs = LeastConfidence(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.model_change, True)

    def test_lcb_mc(self):
        qs_kwargs = {'model_change': True}
        qs = LeastConfidenceBias(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.model_change, True)

    def test_lcb2_mc(self):
        qs_kwargs = {'model_change': True}
        qs = LeastConfidenceDynamicBias(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.model_change, True)

    def test_id(self):
        qs1 = LeastConfidence(model_change=False)
        qs2 = Density()
        qs = CombinedSampler(qs1, qs2, beta=3)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.beta, 3)

    def test_limit_draws(self):
        qs = Random()
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_x, self.test_x,
                        self.train_y, self.test_y,
                        ndraws=self.ndraws-1)
        self.assertEqual(scores.shape, (self.ndraws-1,))


if __name__ == '__main__':
    unittest.main()
