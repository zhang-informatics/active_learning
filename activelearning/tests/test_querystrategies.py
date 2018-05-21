import unittest
import copy
import numpy as np

from sklearn import __version__
sklearn_ver = __version__

from activelearning.querystrategies import *
from activelearning.activelearning import Data


class QueryStrategyTest(unittest.TestCase):
    class DummyClassifier(object):
        def decision_function(self, X):
            return np.array([0.3, -0.3, 0.5, -0.5])
        def predict_proba(self, X):
            return np.array([[0.1, 0.9],
                             [0.3, 0.7],
                             [0.5, 0.5],
                             [0.7, 0.3],
                             [0.9, 0.1]])
    def setUp(self):
        np.random.seed(420)
        XU = np.linspace(1, 15, 15).reshape(5, 3)
        yU = np.array([0,1,0,1,0])
        XL = np.linspace(9, 1, 9).reshape(3, 3)
        yL = np.array([1,0,1])
        self.unlabeled = Data(x=XU, y=yU)
        self.labeled = Data(x=XL, y=yL)
        self.clf = self.DummyClassifier()
        self.args = [self.unlabeled, self.labeled, self.clf]

    def tearDown(self):
        del self.unlabeled
        del self.labeled
        del self.clf

    def test_random(self):
        strategy = Random()
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 1)

    def test_simple_margin(self):
        strategy = SimpleMargin()
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 0)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_margin(self):
        strategy = Margin()
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 2)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_entropy(self):
        strategy = Entropy()
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 2)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_lc(self):
        strategy = LeastConfidence()
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 2)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_lcb(self):
        strategy = LeastConfidenceBias()
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 2)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_lcb2(self):
        strategy = LeastConfidenceDynamicBias()
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 2)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_d2c(self):
        qs_kwargs = {'metric': 'euclidean'}
        strategy = DistanceToCenter(**qs_kwargs)
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 4)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_density(self):
        qs_kwargs = {'metric': 'euclidean'}
        strategy = Density(**qs_kwargs)
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 0)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_density_nan(self):
        XU = np.linspace(1, 15, 15).reshape(5, 3)
        XU[3] = np.array([0., 0., 0.])
        yU = np.array([0,1,0,1,0])
        XL = np.linspace(9, 1, 9).reshape(3, 3)
        yL = np.array([1,0,1])
        unlabeled = Data(x=XU, y=yU)
        labeled = Data(x=XL, y=yL)
        clf = self.DummyClassifier()
        args = [unlabeled, labeled, clf]
        qs_kwargs = {'metric': 'cosine'}
        strategy = Density(**qs_kwargs)
        with self.assertRaises(ValueError, msg="Distances contain NaN values. Check that input vectors != 0."):
            choice = strategy.query(*args)

    def test_minmax_metric(self):
        strategy = MinMax()
        self.assertEqual(strategy.distance_metric, 'euclidean')

    def test_minmax_query_euc(self):
        qs_kwargs = {'metric': 'euclidean'}
        strategy = MinMax(**qs_kwargs)
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 4)
        self.assertEqual(strategy.scores.shape[0], self.unlabeled.x.shape[0])
        self.assertNotIn(np.NaN, strategy.scores)

    def test_minmax_query_mah(self):
        qs_kwargs = {'metric': 'mahalanobis'}
        strategy = MinMax(**qs_kwargs)
        choice = strategy.query(*self.args)
        # A difference in sklearn versions cause a different result to be computed.
        if sklearn_ver == "0.19.0":
            self.assertEqual(choice, 0)
        elif sklearn_ver == "0.18.1":
            self.assertEqual(choice, 4)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_minmax_query_mah_singular(self):
        qs_kwargs = {'metric': 'mahalanobis'}
        XU = np.array([[5,5,5,5,5]] * 5)
        XL = np.array([[3,3,3,3,3]] * 3)
        yU = np.array([0,1,0,1,0])
        yL = np.array([1,0,1])
        U = Data(x=XU, y=yU)
        L = Data(x=XL, y=yL)
        args = [U, L, self.clf]
        strategy = MinMax(**qs_kwargs)
        choice = strategy.query(*args)
        self.assertEqual(choice, 0)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_minmax_LDS(self):
        x1, y1 = np.random.normal(1, 0.1, 20), np.random.normal(3, 0.15, 20)
        x2, y2 = np.random.normal(3, 0.2, 20), np.random.normal(1, 0.1, 20)
        x3, y3 = np.random.normal(2, 1, 5), np.random.normal(2, 1, 5)
        XU = np.dstack((np.concatenate((x1, x2, x3)),
                        np.concatenate((y1, y2, y3))))[0]
        XL = copy.deepcopy(XU)
        YU = np.random.randint(2, size=XU.shape[0])
        YL = np.random.randint(2, size=XL.shape[0])
        self.unlabeled.x = XU
        self.unlabeled.y = YU
        self.labeled.x = XL
        self.labeled.y = YL
        U_orig = copy.deepcopy(self.unlabeled)
        qs_kwargs = {'LDS': True, 'LDS_k': 8, 'LDS_threshold': 4}
        strategy = MinMax(**qs_kwargs)
        choice = strategy.query(self.unlabeled, self.labeled, self.clf)
        self.assertTrue((U_orig.x.shape != self.unlabeled.x.shape))
        self.assertTrue(self.unlabeled.y.shape[0] == self.unlabeled.x.shape[0])

    def test_combined(self):
        qs1 = Entropy()
        qs2 = MinMax()
        beta = 1
        strategy = CombinedSampler(qs1=qs1, qs2=qs2, beta=beta)
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 3)
        self.assertNotIn(np.NaN, strategy.scores)

    def test_combined_dynamic_beta(self):
        qs1 = Entropy()
        qs2 = MinMax()
        beta = 'dynamic'
        strategy = CombinedSampler(qs1=qs1, qs2=qs2, beta=beta)
        choice = strategy.query(*self.args)
        self.assertEqual(choice, 3)
        self.assertEqual(strategy.beta, 'dynamic')
        self.assertNotIn(np.NaN, strategy.scores)

    def test_combined_alpha_float(self):
        qs1 = Entropy()
        qs2 = MinMax()
        beta = 'dynamic'
        alpha = 2
        strategy = CombinedSampler(qs1=qs1, qs2=qs2, beta=beta, alpha=alpha)
        self.assertEqual(strategy.alpha, 2)

    def test_combined_alpha_auto(self):
        qs1 = Entropy()
        qs2 = MinMax()
        beta = 'dynamic'
        alpha = 'auto'
        strategy = CombinedSampler(qs1=qs1, qs2=qs2, beta=beta, alpha=alpha)
        self.assertEqual(strategy.alpha, 'auto')

    def test_combined_alpha_warn(self):
        qs1 = Entropy()
        qs2 = MinMax()
        beta = 1
        alpha = 3
        with self.assertWarns(RuntimeWarning,
                              msg="Alpha set but not used when beta != dynamic"):
            strategy = CombinedSampler(qs1=qs1, qs2=qs2, beta=beta, alpha=alpha)
