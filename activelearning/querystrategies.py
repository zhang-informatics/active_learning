import sys
import warnings
import numpy as np
import scipy.spatial.distance as spd

from collections import namedtuple
from sklearn.neighbors import NearestNeighbors

from .containers import Data

__all__ = ['UncertaintySampler',
           'CombinedSampler',
           'Random',
           'SimpleMargin',
           'Margin',
           'Entropy',
           'LeastConfidence',
           'LeastConfidenceBias',
           'LeastConfidenceDynamicBias',
           'DistanceToCenter',
           'MinMax',
           'Density']


class QueryStrategy(object):
    '''
    Base query strategy class. In general, a query strategy
    consists of a scoring function, which assigns scores to 
    each unlabeled instance, and a query function, which chooses
    and instance from these scores.
    '''
    def __init__(self):
        #  Unlabeled data, labeled data, classifier.
        self.Args = namedtuple('Args', ['U', 'L', 'clf'])
        self.scores = 0

    def get_args(self, *args):
        '''
        Creates a namedtuple instance containing arguments to to score().
            *args is:
        :param Data unlabeled: Unlabeled set.
        :param Data labeled: Labeled set.
        :param sklearn.base.BaseEstimator classifier: Classifier to use. 
        :returns: Arguments
        :rtype: namedtuple
        '''
        if len(args) != 3:
            raise ValueError("Number of arguments must be 3")
        args = self.Args(U=args[0], L=args[1], clf=args[2]) 
        return args

    def score(self, *args, **kwargs):
        '''
        Computes an array of scores for members of U from which
        to choose. 
        '''
        raise NotImplementedError()

    def query(self, *args, **kwargs):
        '''
        Chooses a member from U from scores as computed by self.scores.
        '''
        raise NotImplementedError()


class UncertaintySampler(QueryStrategy):

    def __init__(self, model_change=False):
        super().__init__() 
        self.model_change = model_change
        self.chosen_index = None  # Used by model_change

    def __score(self):
        '''
        In uncertainty sampling it is possible to use model change,
        which is implemented as a wrapper around the scoring function.
        See model_change() below. The __init__() method for a child of
        this class should define the score() method in the following manner:

            if self.model_change is True:
                self.score = self.model_change_wrapper(self.__score)
            else:
                self.score = self.__score
        '''
        raise NotImplementedError()

    def model_change_wrapper(self, score_func):
        '''
        Model change wrapper around the scoring function. See doc
        for __score() above for usage insructions.

        score_mc(x) = score(x, t) - w_o * score(x, t-1)
            where score(x, t) = the score at time t
                                according to the scoring function.
                  score(x, t-1) = the score at the previous time step.
                  w_o = 1 / |L|

        :param function score_func: Scoring function to wrap.
        :returns: Wrapped scoring function.
        :rtype: function
        '''
        def wrapper(*args):
            tmp_args = self.get_args(*args)
            prev_scores = np.empty_like(self.scores)
            np.copyto(prev_scores, self.scores)
            w_o = 1 / tmp_args.L.y.shape[0]
            if self.chosen_index is not None:
                prev_scores = np.delete(self.scores, self.chosen_index, axis=0)
            scores = score_func(*args)
            scores = scores - (w_o * prev_scores)
            return scores
        return wrapper

    def query(self, *args, **kwargs):
        '''
        Chooses a member from U from scores as computed by self.scores.
        :returns: Index from scores.
        :rtype: int
        '''
        self.scores = self.score(*args, **kwargs)
        self.chosen_index = np.argmax(self.scores)
        return self.chosen_index


class CombinedSampler(QueryStrategy):
    '''
    Allows one sampler's scores to be weighted by anothers according
    to the equation:

        score = score_qs1(x) * score_qs2(x)**beta

    N.B. Assumes x* = argmax(score)
    '''
    def __init__(self, qs1=None, qs2=None, beta=1, alpha='auto'):
        '''
        :param QueryStrategy qs1: Main query strategy.
        :param QueryStrategy qs2: Query strategy to use as weight.
        :param float beta: Scale factor for score_qs2.
        :param str|float alpha: Scale factor for beta. Default |L|/|U_0|
        '''
        if qs1 is None or qs2 is None:
            raise ValueError("Must supply both qs1 and qs2")
        super().__init__()
        self.qs1 = qs1
        self.qs2 = qs2
        if beta == 'dynamic':
            self.beta = beta
        else:
            self.beta = float(beta)
        if alpha == 'auto':
            self.alpha = alpha
        else:
            self.alpha = float(alpha)
            if self.beta != 'dynamic':
                warnings.warn("Alpha set but not used when beta != dynamic",
                              RuntimeWarning)
        self.U_0_size = None

    def __str__(self):
        return "Combined Sampler: qs1: {0}; qs2 {1}".format(str(self.qs1),
                                                            str(self.qs2))

    # TODO: Remove. No longer used.
    def _compute_alpha(self, *args):
        '''
        alpha = |L|/|U_0|
            where U_0 is the size of the initial unlabeled set.
        :returns: alpha
        :rtype: float
        '''
        args = self.get_args(*args)
        if self.U_0_size is None:
            self.U_0_size = args.U.x.shape[0]
        alpha = args.L.x.shape[0] / self.U_0_size
        return alpha

    def _compute_beta(self, *args):
        '''
        Dynamic beta is computed according to the ratio of number of labeled
        to unlabeled samples.

        beta = |U|/|L|
        :returns: beta
        :rtype: float        
        '''
        args = self.get_args(*args)
        if self.alpha == 'auto':
            alpha = self._compute_alpha(*args)
        else:
            alpha = self.alpha
#        beta = (args.U.x.shape[0] / args.L.x.shape[0])
        beta = 2 * (args.U.x.shape[0] / args.L.x.shape[0])
        return beta

    def _normalize_scores(self, scores):
        '''
        Computes minmax normalization to map scores to the (0,1) interval.
        '''
        if all(scores == scores[0]):  # If all scores are equal.
            norm_scores = scores if scores[0] < 1 else (1/scores[0]) 
        else:
            num = scores - np.min(scores)
            denom = np.max(scores) - np.min(scores)
            norm_scores = num / denom
        return norm_scores

    def score(self, *args):
        '''
        Computes the combined scores from qs1 and qs2.
        :returns: scores
        :rtype: numpy.ndarray
        '''
        if self.beta == 'dynamic':
            beta = self._compute_beta(*args)
        else:
            beta = self.beta
        qs1_scores = self._normalize_scores(self.qs1.score(*args))
        qs2_scores = self._normalize_scores(self.qs2.score(*args))
        scores = qs1_scores * (qs2_scores**beta) 
        return scores

    def query(self, *args, **kwargs):
        '''
        Chooses a member from U from scores as computed by self.scores.
        :returns: Index from scores.
        :rtype: int
        '''
        self.scores = self.score(*args, **kwargs)
        index = np.argmax(self.scores)
        return index
        

class Random(QueryStrategy):
    '''
    Random query strategy. Equivalent to passive learning.
    '''
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Random Sampler"

    def score(self, *args):
        '''
        In the random case, scores are just the max index to choose from.
        '''
        args = self.get_args(*args)
        return args.U.x.shape[0]

    def query(self, *args, **kwargs):
        '''
        Chooses a member from U from scores as computed by self.scores.
        :returns: Random index in range(self.scores).
        :rtype: int
        '''
        self.scores = self.score(*args)
        index = np.random.choice(self.scores)
        return index


class SimpleMargin(QueryStrategy):
    '''
    Finds the example x that is closest to the separating hyperplane.
    '''
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Simple Margin Sampler"
    
    def score(self, *args):
        '''
        Computes distances to the hyperplane for each member of
        the unlabeled set.
        '''
        args = self.get_args(*args)
        distances = args.clf.decision_function(args.U.x)
        scores = np.abs(distances)
        return scores
        
    def query(self, *args, **kwargs):
        '''
        Chooses the member from the unlabeled set with the minimum distance
        to the hyperplane.
        :returns: Index from the unlabeled set.
        :rtype: int
        '''
        self.scores = self.score(*args, **kwargs)
        index = np.argmin(self.scores) 
        return index


class Margin(QueryStrategy):
    '''
    Margin Sampler. Chooses the member from the unlabeled set
    with the smallest difference between the posterior probabilities
    of the two most probable labels.

    x = argmin_x P_clf(y_hat1|x) - P_clf(y_hat2|x)
        where y_hat1 is the most probable label
          and y_hat2 is the second most probable label.

    '''
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Margin Sampler"

    def score(self, *args):
        '''
        Computes the difference between posterior probability estimates
        for the top two most probable labels.
        :returns: Posterior probability differences.
        :rtype: numpy.ndarray
        '''
        args = self.get_args(*args)
        probs = args.clf.predict_proba(args.U.x)
        # Sort each row from high to low. Multiply by -1 to keep sign the same.
        probs = np.sort(-probs, axis=1) * -1 
        # Compute the difference between first and second most likely labels.
        scores = probs[:,0] - probs[:,1]
        return scores
         
    def query(self, *args, **kwargs):
        '''
        Chooses the member from the unlabeled set with the smallest difference
        between posterior probabilities.
        :returns: Index from the unlabeled set.
        :rtype: int
        '''
        # Index is the pair with the smallest difference.
        self.scores = self.score(*args, **kwargs)
        index = np.argmin(self.scores)
        return index


class Entropy(UncertaintySampler):
    '''
    Entropy Sampler. Chooses the member from the unlabeled set
    with the greatest entropy across possible labels.

    x = argmax_x -sum_i (P_clf(y_i|x) * log2(P_clf(y_i|x)))
    '''
    def __init__(self, model_change=False):
        super().__init__(model_change=model_change)
        # Define self.score()
        if self.model_change is True:
            self.score = self.model_change_wrapper(self.__score)
        else:
            self.score = self.__score

    def __str__(self):
        return "Entropy Sampler"

    def __score(self, *args):
        '''
        Computes entropies for each member of the unlabeled set.
        :returns: Entropies.
        :rtype: numpy.ndarray
        '''
        args = self.get_args(*args)
        probs = args.clf.predict_proba(args.U.x)
        scores = -np.sum(np.multiply(probs, np.log2(probs)), axis=1)
        return scores


class LeastConfidence(UncertaintySampler):
    '''
    Least confidence (uncertainty sampling). Chooses the member from
    the unlabeled set with the greatest uncertainty, i.e. the greatest
    posterior probability of all labels except the most likely one.

    x = argmax_x 1 - P_clf(y_hat|x)
        where y_hat = argmax_y P_clf(y|x)
    '''
    def __init__(self, model_change=False):
        super().__init__(model_change=model_change)
        # Define self.score()
        if self.model_change is True:
            self.score = self.model_change_wrapper(self.__score)
        else:
            self.score = self.__score

    def __str__(self):
        return "Least Confidence"

    def __score(self, *args):
        '''
        Computes leftover probabilities for each member of the unlabeled set.
        :returns: Leftover probabilities.
        :rtype: numpy.ndarray
        '''
        args = self.get_args(*args)
        probs = args.clf.predict_proba(args.U.x)
        scores = 1 - np.max(probs, axis=1)
        return scores
    

class LeastConfidenceBias(UncertaintySampler):
    '''
    Least confidence with bias. This is the same as least confidence, but
    moves the decision boundary according to the current class distribution. 

    x = argmax_x -| P_clf(y_hat|x) / P_max         if P_clf(y_hat|x) < P_max
                  | (1 - P_clf(y_hat|x)) / P_max   otherwise
            where P_max = mean(0.5, 1 - pp)
                where pp = % of positive examples in labeled_set.
    '''
    def __init__(self, model_change=False):
        super().__init__(model_change)
        # Define self.score()
        if self.model_change is True:
            self.score = self.model_change_wrapper(self.__score)
        else:
            self.score = self.__score

    def __str__(self):
        return "Least Confidence with Bias"

    def __score(self, *args):
        '''
        Computes leftover probabilities for each member of the unlabeled set, 
        adjusted for the current class distribution.
        :returns: scores
        :rtype: numpy.ndarray
        '''
        args = self.get_args(*args)
        pp = sum(args.L.y) / args.L.y.shape[0]
        p_max = np.mean([0.5, 1 - pp])
        probs = np.max(args.clf.predict_proba(args.U.x), axis=1)
        scores = np.where(probs < p_max,        # If
                          probs / p_max,        # Then
                          (1 - probs) / p_max)  # Else
        return scores


class LeastConfidenceDynamicBias(UncertaintySampler):
    '''
    Least confidence with dynamic bias. This is the same as least confidence
    with bias, but the bias also adjusts for the relative sizes of the
    labeled and unlabeled data sets.

    x = argmax_x -| P_clf(y_hat|x) / P_max         if P_clf(y_hat|x) < P_max
                  | (1 - P_clf(y_hat|x)) / P_max   otherwise
            where P_max = w_b * (1 - pp) + w_u * 0.5
              where pp = % of positive examples in labeled_set.
                    w_b = 1 - w_u
                  where w_u = |L| / |U_0|
                    where U_0 = the initial unlabeled set.

    '''
    def __init__(self, model_change=False):
        super().__init__(model_change)
        self.U_0_size = -1
        # Define self.score()
        if self.model_change is True:
            self.score = self.model_change_wrapper(self.__score)
        else:
            self.score = self.__score

    def __str__(self):
        return "Least Confidence with Dynamic Bias"

    def __score(self, *args):
        '''
        :returns: scores
        :rtype: numpy.ndarray
        '''
        args = self.get_args(*args)
        if self.U_0_size < 0:  # Set U_0_size if unset (-1)
            self.U_0_size = args.U.x.shape[0]
        pp = sum(args.L.y) / args.L.y.shape[0]
        w_u = args.L.y.shape[0] / self.U_0_size
        w_b = 1 - w_u
        p_max = w_b * (1 - pp) + w_u * 0.5
        probs = args.clf.predict_proba(args.U.x)[:, 1]
        scores = np.where(probs < p_max,        # If
                          probs / p_max,        # Then
                          (1 - probs) / p_max)  # Else
        return scores


class DistanceToCenter(QueryStrategy):
    '''
    Distance to Center sampling. Measures the distance of each point
    to the average X (center) in the labeled data set.

    x* = argmin_x 1 / (1 + dist(x, x_L))
        where dist(A, B) is the distance between vectors A and B.
              x_L is the mean vector in L (L's center).
    '''
    def __init__(self, metric='euclidean'):
        '''
        :param str metric: Distance metric to use. See spd.cdist doc for
                           available metrics.
        '''
        super().__init__()
        self.distance_metric = metric
        self.VI = None

    def __str__(self):
        return "Distance to Center Sampler"

    def score(self, *args):
        '''
        :returns: Distances.
        :rtype: numpy.ndarray
        '''
        args = self.get_args(*args)
        mean_labeled_x = np.mean(args.L.x, axis=0)
        if self.distance_metric == 'mahalanobis' and self.VI is None:
            full_matrix = np.vstack([args.U.x, args.L.x]).T
            # Use pseudo inverse because features are sparse.
            self.VI = np.linalg.pinv(np.cov(full_matrix)).T
        distances = spd.cdist([mean_labeled_x], args.U.x,
                              metric=self.distance_metric, VI=self.VI)
        densities = 1 / (1 + distances)
        return densities[0]

    def query(self, *args, **kwargs):
        '''
        Chooses the member from the unlabeled set with the shortest distance
        from the average labeled instance.
        :returns: Index from the unlabeled set.
        :rtype: int
        '''
        # Index is the pair with the smallest difference.
        self.scores = self.score(*args, **kwargs)
        index = np.argmin(self.scores)
        return index


class MinMax(QueryStrategy):
    '''
    Finds the exmaple x in U that has the maximum smallest distance
    to every point in L. Ensures representative coverage of the dataset.

    x* = argmax_xi (min_xj dist(xi, xj))
        where xi in U
              xj in L
              dist(.) is the given distance metric.
    '''
    def __init__(self, metric='euclidean', LDS=False, LDS_k=30,
                 LDS_threshold="auto"):
        '''
        :param str metric: Distance metric to use. See spd.cdist doc for
                           available metrics.
        '''
        super().__init__()
        self.distance_metric = str(metric)
        self.VI = None
        if LDS is True:
            self.use_LDS = True
            self.first_run = True
            self.k = int(LDS_k)
            self.threshold = LDS_threshold
        else:
            self.use_LDS = False
        
    def __str__(self):
        return "Min Max Sampler"

    def _nearest_neighbors(self, unlabeled_x, k=30):
        neighbors = {}
        # Because we are fitting NNs to themselves,
        # we'll have to get rid of the first NN. 
        cov = np.cov(unlabeled_x)
        inv_cov = np.linalg.pinv(cov).T
        NNs = NearestNeighbors(n_neighbors=k, algorithm='brute',
                               metric='mahalanobis',
                               metric_params={'VI': inv_cov})
                               #metric_params={'V': cov, 'VI': inv_cov})
        NNs.fit(unlabeled_x)
        neighbor_indices = NNs.kneighbors(return_distance=False)
        num_samples = neighbor_indices.shape[0]
        neighbors = dict(zip(range(num_samples), neighbor_indices))
        return neighbors

    def _compute_LDS(self, neighbors, unlabeled_x):
        def weight(x_i, x_j):
            weight = len(set(neighbors[x_i]) & set(neighbors[x_j]))
            return weight
        num_Ux = unlabeled_x.shape[0]
        LDS = np.zeros(num_Ux, dtype=np.float32)
        for x_i in range(num_Ux):
            nns = neighbors[x_i]
            LDS[x_i] = np.sum([weight(x_i, x_j) for x_j in nns]) / len(nns)
        return LDS

    def _get_subset(self, LDS_scores, threshold, unlabeled_set):
        keep_indices = [i for (i, score) in enumerate(LDS_scores)
                        if score >= threshold]
        return unlabeled_set[keep_indices]

    def init_LDS(self, *args):
        '''
        Computes nearest neighbors for all points in U.x.
        Computes the Local Density Score LDS for each point in U.x.

            LDS(x_i) = \frac{sum_{x_j \in NN(x_i)} weight(x_i, x_j)} / k
                where weight(x_i, x_j) = |NN(x_i) \cap NN(x_j)|
            
        Computes the subset of points in U.x for which LDS(x) > k/2
        N.B. changes U.x and U.y permanently.
        '''
        args = self.get_args(*args)
        neighbors = self._nearest_neighbors(args.U.x, k=self.k) 
        scores = self._compute_LDS(neighbors, args.U.x)
        if self.threshold == "auto":
            threshold = (self.k / 2)
        else:
            threshold = self.threshold
        x, y = self._get_subset(scores, threshold, args.U)
        print("""
        LDS: k {0},
        threshold {1},
        original data size {2},
        subset size {3}""".format(self.k, threshold,
                                  args.U.x.shape[0],
                                  x.shape[0])
             )
        args.U.x = x
        args.U.y = y

    # TODO: Precompute distances to avoid redundant computation.
    def score(self, *args):
        '''
        Computes minimum distance between each member of unlabeled_x
            and each member of labeled_x.
        :returns: Minimum distances from each unlabeled_x to each labeled_x.
        :rtype: numpy.ndarray
        '''
        args = self.get_args(*args)
        if self.use_LDS is True:
            if self.first_run is True:
                self.init_LDS(*args)  # Modifies args.U
                self.first_run = False
        if self.distance_metric == 'mahalanobis' and self.VI is None:
            full_matrix = np.vstack([args.U.x, args.L.x]).T
            # Use pseudo inverse because features are sparse.
            self.VI = np.linalg.pinv(np.cov(full_matrix)).T
        distances = spd.cdist(args.U.x, args.L.x,
                              metric=self.distance_metric, VI=self.VI)
        min_dists = np.min(distances, axis=1)
        return min_dists

    def query(self, *args, **kwargs):
        '''
        Chooses the member from the unlabeled set with the largest minimum
        distances to every point in the labeled set.
        :returns: Index from the unlabeled set.
        :rtype: int
        '''
        self.scores = self.score(*args, **kwargs) 
        index = np.argmax(self.scores)     
        return index


class Density(QueryStrategy):
    '''
    Finds the example x in U that has the greatest average distance to
    every other point in U. 

    x* = argmin_x 1/U \sum_{u=1} ( 1 /(1 + dist(x, x_u)) )
    '''
    def __init__(self, metric='euclidean'):
        '''
        :param str metric: Distance metric to use. See spd.cdist doc for
                           available metrics.
        '''
        super().__init__()
        self.distance_metric = metric
        self.VI = None

    def __str__(self):
        return "Density Sampler"

    def score(self, *args):
        '''
        Computes average distance between each member of U and each other
        member of U.
        :returns: Minimum distances from each point in U to each other point.
        :rtype: numpy.ndarray
        '''
        args = self.get_args(*args)
        # Computing similarity to itself will fail. 
        if args.U.x.shape[0] == 1:
            return np.empty(1)
        if self.distance_metric == 'mahalanobis' and self.VI is None:
            full_matrix = np.vstack([args.U.x, args.L.x]).T
            # Use pseudo inverse because features are sparse.
            self.VI = np.linalg.pinv(np.cov(full_matrix)).T
        distances = spd.cdist(args.U.x, args.U.x,
                              metric=self.distance_metric, VI=self.VI) 
        if np.isnan(distances).any():
            raise ValueError("Distances contain NaN values. Check that input vectors != 0.")
        num_x = args.U.x.shape[0]
        # Remove zero scores b/c we want distance from every OTHER point.
        np.fill_diagonal(distances, np.NaN)
        distances = distances[~np.isnan(distances)].reshape(num_x, num_x - 1)
        similarities = 1 / (1 + distances)
        mean_sims = np.mean(similarities, axis=1)
        return mean_sims

    def query(self, *args, **kwargs):
        '''
        Chooses the member from the unlabeled set with the smallest average
        distance to every other member of the unlabeled set.
        :returns: Index from the unlabeled set.
        :rtype: int
        '''
        self.scores = self.score(*args, **kwargs) 
        index = np.argmin(self.scores)     
        return index
