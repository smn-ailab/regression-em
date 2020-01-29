"""Package to perform regression EM algorithm self.
For the detail of this algortithm,
see [https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf].
"""
from random import random
from typing import Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import BaseEstimator

# Define type.
RegParam = Tuple[np.ndarray, float]


class RegressionEM(BaseEstimator):
    """Regression EM can estimate latent factors based on below model.

       Outcome = Left latent * Right latent

       For the detail of Regression EM algortithm,
       see [https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf].
       "Position Bias Estimation for Unbiased Learning to Rank in Personal Search"
       Xuanhui Wang et al.

       :Parameters:

       alpha : float, default=1
       Weights associated with regularization term

       max_iter : int, default=100
       Maximum number of iterations taken for the solvers to converge

       epsilon : float, default=1e-10
       Tiny value for avoiding zero devide

       with_sample_weights : bool, default = False
       Indicator to use imbalance weight

       :Example:

       >>> import RegressionEM
       >>> left_feat = array([[0.17843638, 0.09311004, 0.89600447, ...],
       [0.55349066, 0.83427622, 0.34841103, ...],
       ...,
       [0.22199485, 0.19540406, 0.02678277, ...],
       [0.62612729, 0.71996384, 0.66445362, ...]])
       >>> right_feat = array([[..., 0.6177678 , 0.69322733, 0.95146727],
       [..., 0.96681348, 0.79037145, 0.45834361],
       ...,
       [..., 0.64773992, 0.86541352, 0.04755084],
       [..., 0.37910497, 0.44344932, 0.48168189]])
       >>> X = np.hstack([left_feat, right_feat])
       >>> X = array([[0.17843638, 0.09311004, 0.89600447, ..., 0.6177678 , 0.69322733, 0.95146727],
       [0.55349066, 0.83427622, 0.34841103, ..., 0.96681348, 0.79037145, 0.45834361],
       ...,
       [0.22199485, 0.19540406, 0.02678277, ..., 0.64773992, 0.86541352, 0.04755084],
       [0.62612729, 0.71996384, 0.66445362, ..., 0.37910497, 0.44344932, 0.48168189]])
       >>> y = array([False, False, False, ...,  True, False,  True])
       >>> rem = RegressionEM(max_iter=100, with_sample_weights=True, alpha=1)
       >>> rem.fit(X, y, 100)
    """

    def __init__(self, alpha: float = 0, max_iter: int = 100, epsilon: float = 10 ** -10, with_sample_weights: bool = False) -> None:
        """Initialize hyper parameters."""
        self._alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.with_sample_weights = with_sample_weights

    @staticmethod
    def calc_probs(coef: np.ndarray, intercept: float, feat_vec: np.ndarray) -> np.ndarray:
        """Return probabilities calculated based on the definition of logistic regression.

        :param coef: Coefficient vector of LR.
        :param intercept: Intercept parameter of LR.
        :param feat_vec: Feature vector.
        :return: Sequence of probabilities.
        """
        # 1 / exp(- (feature @ coef + intercept))
        return 1 / (1 + np.exp(- feat_vec @ coef - intercept))

    def calc_logits(self, probs: np.ndarray) -> np.ndarray:
        """Return logits calculated from probability array.

        :param probs: Probability array to be converted to logists.
        :param epsilon: Tiny value for clipping.
        :return: Logist arrray.
        """
        # Perform clipping to avoid log(0) and zero division.
        clipped = probs
        clipped[clipped == 0] = self.epsilon
        clipped[clipped == 1] = 1 - self.epsilon

        return np.log(clipped / (1 - clipped))

    @staticmethod
    def calc_responsibility(target_prob: float, ref_prob: float, is_positive: bool) -> float:
        """Return responsibility to be used in EM algorithm.
        For detail, see eq.1 of [https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf].

        :param target_prob: Probability calculated from the parameters to be updated.
        :param ref_prob: Reference probability.
        :param is_positive: If True, the sample has a positive label.
        :return: Responsibility.
        """

        if is_positive or target_prob == ref_prob == 1:
            return 1.0
        else:
            return (target_prob * (1 - ref_prob)) / (1 - target_prob * ref_prob)

    def update_responsibilities(self, target_params: RegParam, target_feat: np.ndarray,
                                ref_params: RegParam, ref_feat: np.ndarray,
                                labels: Sequence[bool]) -> np.ndarray:
        """Return responsibilities based on M-step parameters.

        .. Note::
            target_params and ref_params must be (weight vector, intercept).

        :param target_params: Regression params corresponding to the latent factor to be updated.
        :param target_feat: Feature matrix corresponding to the latent factor to be updated.
        :param ref_params: Regression params corresponding to the refered latent factor.
        :param ref_feat: Feature matrix corresponding to the refered latent factor.
        :param labels: Sequence of boolean indicating each sample is positivei or negative.
        :return: List of responsibilities.
        """
        # The format of params must be (coef vector, intercept).
        # Calculating ref prob. with updated ref params for updating target param

        target_probs = self.calc_probs(target_params[0], target_params[1], target_feat)
        ref_probs = self.calc_probs(ref_params[0], ref_params[1], ref_feat)

        return np.vectorize(self.calc_responsibility)(target_probs, ref_probs, labels)

    def update_params(self, feat_mat: np.ndarray, responsibilities: np.ndarray, sample_weights) -> Tuple[RegParam, float]:
        """Return fitted Logistic Regression params.
        For detail, see eq.2 of [https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf].

        :param feat_mat: Feature matrix to be used to learn responsibility.
        :param responsibilities: Sequence of responsibilities calculated at E-step.
        :param sample_weights: Sequence of Weights for treating imbalance dataset.
        for positive data: 1/n-positive data
        for negative data: 1/n-negative data
        :return: Updated M-step params.
        """
        if self._alpha:
            reg = Ridge(alpha=self._alpha)
        else:
            reg = LinearRegression()
        reg.fit(feat_mat, self.calc_logits(responsibilities), sample_weights)
        return reg.coef_, reg.intercept_

    def _calc_log_likelihood(self, left_feat: np.ndarray, right_feat: np.ndarray,
                             labels: np.ndarray) -> np.ndarray:
        """Return log likelihood.
        positive label: log(outcome_probs)
        negative label: log(1-outcome_probs)

        :param left_feat: Feature matrix to be used to learn left params.
        :param right_feat: Feature matrix to be used to learn  parightrams.
        :param labels: Sequence of boolean indicating each sample is positivei or negative.
        :return: log_likelihood
        """
        # Calculate predicted probabilities of outcome.
        outcome_probs = self.calc_probs(self.left_params[0], self.left_params[1], left_feat) * \
            self.calc_probs(self.right_params[0], self.right_params[1], right_feat)

        # Calculate log likelihood with positive samples only.
        positive_sample_probs = outcome_probs[labels]  # Get list of probs whose sample labels are True.
        positive_sample_probs[positive_sample_probs == 0] = self.epsilon  # Clip probs to avoid log(0).
        positive_sample_log_lh = np.sum(np.log(positive_sample_probs))

        # Apply the same procedure to negative ones.
        negative_sample_probs = outcome_probs[~labels]
        negative_sample_probs[negative_sample_probs == 1] = 1 - self.epsilon
        negative_sample_log_lh = np.sum(np.log(1 - negative_sample_probs))

        return positive_sample_log_lh + negative_sample_log_lh

    # def fit(self, left_feat: np.ndarray, right_feat: np.ndarray, labels: np.ndarray) -> None:
    def fit(self, X, y, index: int) -> None:
        """Estimate regression EM params.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_left and right latent features)
                  Feature matrix derived from concatenating left latent features with right latent features.
        :param y: array-like of shape (n_samples,)
                  Sequence of labels indicating each sample is positive or negative.
        :param index: int
                      the index to devide X into left latent features and right latent features.
        """
        # separate dataset with index.
        left_feat, right_feat = np.hsplit(X, [index])

        # Initialize params (feature weight, intercept).
        self.left_params = (np.random.rand(left_feat.shape[1]), random())
        self.right_params = (np.random.rand(right_feat.shape[1]), random())
        self.log_likelihoods = [self._calc_log_likelihood(left_feat, right_feat, y)]
        max_ll = self.log_likelihoods[-1]
        best_left_params = self.left_params
        best_right_params = self.right_params

        sample_weights = None
        if self.with_sample_weights:
            pos_ratio = np.count_nonzero(y) / y.size
            sample_weights = [1 / pos_ratio if l else 1 / (1 - pos_ratio) for l in y]

        for epoch in range(self.max_iter):
            # Update left latent params
            left_responsibilities = self.update_responsibilities(self.left_params, left_feat, self.right_params, right_feat, y)
            self.left_params = self.update_params(left_feat, left_responsibilities, sample_weights)

            # Update right latent params
            right_responsibilities = self.update_responsibilities(self.right_params, right_feat, self.left_params, left_feat, y)
            self.right_params = self.update_params(right_feat, right_responsibilities, sample_weights)

            # calculating log likelihood and judging convergence
            self.log_likelihoods.append(self._calc_log_likelihood(left_feat, right_feat, y))

            if max_ll < self.log_likelihoods[-1]:
                max_ll = self.log_likelihoods[-1]
                best_left_params = self.left_params
                best_right_params = self.right_params

        self.left_params = best_left_params
        self.right_params = best_right_params

    def predict_proba(self, X, index: int) -> np.ndarray:
        """Return predicted probabilities.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_left and right latent features)
                  Feature matrix derived from concatenating left latent features with right latent features.
        :param index: int
                      the index to devide X into left latent features and right latent features.
        :return: Predicted probabilities.
        """
        # separate dataset with index
        left_feat, right_feat = np.hsplit(X, [index])

        return self.calc_probs(self.left_params[0], self.left_params[1], left_feat) * \
            self.calc_probs(self.right_params[0], self.right_params[1], right_feat)

    def predict(self, X, index: int) -> np.ndarray:
        """Return predicted labels.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_left and right latent features)
                  Feature matrix derived from concatenating left latent features with right latent features.
        :param index: int
                      the index to devide X into left latent features and right latent features.
        :return: Predicted labels.
        """

        # separate dataset
        left_feat, right_feat = np.hsplit(X, [index])

        # calculating probs
        probs = self.predict_proba(left_feat, right_feat)
        return np.array([p >= 0.5 for p in probs])
