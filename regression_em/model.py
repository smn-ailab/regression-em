"""Package to perform Regression EM algorithm.

For the detail of this algortithm,
["Position Bias Estimation for Unbiased Learning to Rank in Personal Search" Xuanhui Wang et al.](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf)
"""
from random import random
from typing import Tuple, Union, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import BaseEstimator


# Define type.
Matrix = Union[np.ndarray, sp.csr_matrix]
RegParam = Tuple[np.array, float]


class RegressionEM(BaseEstimator):
    """Regression EM can estimate latent factors based on below model.

    Outcome = Left latent * Right latent

    For the detail of Regression EM algortithm,
    ["Position Bias Estimation for Unbiased Learning to Rank in Personal Search" Xuanhui Wang et al.](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf)

    :Parameters:

    alpha : float, default=1
    Weights associated with regularization term

    max_iter : int, default=100
    Maximum number of iterations taken for the solvers to converge

    epsilon : float, default=1e-10
    Tiny value to avoid division by zero.

    class_weights : 'balanced', default = None
    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``
    Note that these weights will be multiplied with sample_weight (passed
    through the fit method) if sample_weight is specified.

    split_index : int
    The first column index of right latent feature.

    :Example:

    >>> import RegressionEM
    >>> left_feat = array([[0.17843638, 0.09311004, 0.89600447, ...],
    [0.55349066, 0.83427622, 0.34841103, ...],
    ...,
    [0.22199485, 0.19540406, 0.02678277, ...],
    [0.62612729, 0.71996384, 0.66445362, ...]])
    >>> right_feat = array([[0.6177678 , 0.69322733, 0.95146727, ...],
    [0.96681348, 0.79037145, 0.45834361, ...],
    ...,
    [0.64773992, 0.86541352, 0.04755084, ...],
    [0.37910497, 0.44344932, 0.48168189, ...]])
    >>> X = np.hstack([left_feat, right_feat])
    >>> y = array([0, 0, 0, ...,  1, 0, 1])
    >>> rem = model.RegressionEM(max_iter=10, class_weights='balanced', alpha=1, split_index=100)
    >>> rem.fit(X, y, 100)
    """

    def __init__(self, split_index: int, alpha: float = 0, max_iter: int = 100, epsilon: float = 10 ** -10, class_weights: str = None) -> None:
        """Initialize hyper parameters."""
        self._split_index = split_index
        self._alpha = alpha
        self._max_iter = max_iter
        self._epsilon = epsilon
        self._class_weights = class_weights

    @staticmethod
    def _integers_to_bools(y: np.array) -> np.array:
        """Convert the type of labels from integers to bools."""
        return np.array([True if i > 0 else False for i in y])

    @staticmethod
    def _calc_sample_weights(labels: np.array, class_weights: Optional[str]) -> np.array:
        sample_weights = None
        if class_weights == 'balanced':
            pos_ratio = np.sum(labels) / labels.size
            sample_weights = np.array([1 / pos_ratio if l else 1 / (1 - pos_ratio) for l in labels])
        return sample_weights

    @staticmethod
    def _calc_probs(coef: np.array, intercept: float, feat_mat: Matrix) -> np.array:
        """Return probabilities calculated based on the definition of logistic regression.

        :Parameters:

        coef: Coefficient vector of LR.
        intercept: Intercept parameter of LR.
        feat_vec: Feature vector.

        :return:

        Sequence of probabilities.
        """
        return 1 / (1 + np.exp(- feat_mat @ coef - intercept))

    def _calc_logits(self, probs: np.array) -> np.array:
        """Return logits calculated from probability array.

        :Parameters:

        probs: Probability array to be converted to logists.
        epsilon: Tiny value for clipping.

        :return:

        Logist arrray.
        """
        # Perform clipping to avoid log(0) and zero division.
        if sp.issparse(probs):
            clipped = probs.A
        else:
            clipped = probs
        clipped[clipped == 0] = self._epsilon
        clipped[clipped == 1] = 1 - self._epsilon

        return np.log(clipped / (1 - clipped))

    @staticmethod
    def _calc_responsibility(target_prob: float, ref_prob: float, is_positive: bool) -> float:
        """Return responsibility to be used in EM algorithm.

        For detail, see eq.1 of https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf.

        :Parameters:

        target_prob: Probability calculated from the parameters to be updated.
        ref_prob: Reference probability.
        is_positive: If True, the sample has a positive label.

        :return:

        Responsibility.
        """
        if is_positive or target_prob == ref_prob == 1:
            return 1.0
        else:
            return (target_prob * (1 - ref_prob)) / (1 - target_prob * ref_prob)

    def _update_responsibilities(self, target_params: RegParam, target_feat: Matrix,
                                 ref_params: RegParam, ref_feat: Matrix,
                                 labels: np.array) -> np.array:
        """Return responsibilities based on M-step parameters.

        .. Note::
            target_params and ref_params must be (weight vector, intercept).


        :Parameters:

        target_params: Regression params corresponding to the latent factor to be updated.
        target_feat: Feature matrix corresponding to the latent factor to be updated.
        ref_params: Regression params corresponding to the refered latent factor.
        ref_feat: Feature matrix corresponding to the refered latent factor.
        labels: Sequence of boolean indicating each sample is positive or negative.

        :return:

        List of responsibilities.
        """
        # The format of params must be (coef vector, intercept).
        # Calculating ref prob. with updated ref params for updating target param
        target_probs = self._calc_probs(target_params[0], target_params[1], target_feat)
        ref_probs = self._calc_probs(ref_params[0], ref_params[1], ref_feat)

        return np.vectorize(self._calc_responsibility)(target_probs, ref_probs, labels)

    def _update_params(self, feat_mat: Matrix, responsibilities: np.array, sample_weights: np.array) -> RegParam:
        """Return fitted Logistic Regression params.

        For detail, see eq.2 of https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf.

        :Parameters:

        feat_mat: Feature matrix to be used to learn responsibility.
        responsibilities: Sequence of responsibilities calculated at E-step.
        sample_weights: Sequence of Weights for treating imbalance dataset.
        for positive data: 1/n-positive data
        for negative data: 1/n-negative data

        :return:

        Updated M-step params.
        """
        if self._alpha:
            reg = Ridge(alpha=self._alpha)
        else:
            reg = LinearRegression()

        # Linear Regression performs worse on sparse matrix #13460 Problem
        # https://github.com/scikit-learn/scikit-learn/issues/13460
        # To install pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn

        reg.fit(feat_mat, self._calc_logits(responsibilities), sample_weights)
        return reg.coef_, reg.intercept_

    def _calc_log_likelihood(self, left_params: np.array, left_feat: Matrix, right_params: np.array, right_feat: Matrix,
                             labels: np.array) -> np.array:
        """Return log likelihood.

        positive label: log(outcome_probs)
        negative label: log(1-outcome_probs)

        :Parameters:

        left_feat: Feature matrix to be used to learn left params.
        right_feat: Feature matrix to be used to learn  parightrams.
        labels: Sequence of boolean indicating each sample is positivei or negative.

        :return:

        log_likelihood
        """
        # Calculate predicted probabilities of outcome.
        outcome_probs = self._calc_probs(left_params[0], left_params[1], left_feat) * \
            self._calc_probs(right_params[0], right_params[1], right_feat)

        # Calculate log likelihood with positive samples only.
        positive_sample_probs = outcome_probs[labels]  # Get list of probs whose sample labels are True.
        positive_sample_probs[positive_sample_probs == 0] = self._epsilon  # Clip probs to avoid log(0).
        positive_sample_log_lh = np.sum(np.log(positive_sample_probs))

        # Apply the same procedure to negative ones.
        negative_sample_probs = outcome_probs[np.logical_not(labels)]
        negative_sample_probs[negative_sample_probs == 1] = 1 - self._epsilon
        negative_sample_log_lh = np.sum(np.log(1 - negative_sample_probs))

        return positive_sample_log_lh + negative_sample_log_lh

    def fit(self, X: Matrix, y: np.array) -> None:
        """Estimate regression EM params.

        :Parameters:

        X: {array-like, sparse matrix} of shape (n_samples, n_left and right latent features)
            Feature matrix derived from concatenating left latent features with right latent features.
        y: array-like of shape (n_samples,)
            Sequence of labels indicating each sample is positive or negative.
        """
        # separate dataset with index.
        if sp.issparse(X):
            left_feat = X[:, :self._split_index]
            right_feat = X[:, self._split_index:]
        else:
            left_feat, right_feat = np.hsplit(X, [self._split_index])

        # convert y to boolean
        labels = self._integers_to_bools(y)

        # Initialize params (feature weight, intercept).
        self.left_params = (np.random.rand(left_feat.shape[1]), random())
        self.right_params = (np.random.rand(right_feat.shape[1]), random())

        self.log_likelihoods = [self._calc_log_likelihood(self.left_params, left_feat, self.right_params, right_feat, labels)]
        max_ll = self.log_likelihoods[-1]
        best_left_params = self.left_params
        best_right_params = self.right_params

        sample_weights = self._calc_sample_weights(labels, self._class_weights)

        for epoch in range(self._max_iter):
            # Update left latent params
            left_responsibilities = self._update_responsibilities(self.left_params, left_feat, self.right_params, right_feat, labels)
            self.left_params = self._update_params(left_feat, left_responsibilities, sample_weights)

            # Update right latent params
            right_responsibilities = self._update_responsibilities(self.right_params, right_feat, self.left_params, left_feat, labels)
            self.right_params = self._update_params(right_feat, right_responsibilities, sample_weights)

            # calculating log likelihood and judging convergence
            self.log_likelihoods.append(self._calc_log_likelihood(self.left_params, left_feat, self.right_params, right_feat, labels))

            if max_ll < self.log_likelihoods[-1]:
                max_ll = self.log_likelihoods[-1]
                best_left_params = self.left_params
                best_right_params = self.right_params

        self.left_params = best_left_params
        self.right_params = best_right_params

    def predict_proba(self, X: Matrix) -> np.array:
        """Return predicted probabilities.

        :Parameters:

        X: {array-like, sparse matrix} of shape (n_samples, n_left and right latent features)
            Feature matrix derived from concatenating left latent features with right latent features.

        :return:

        Predicted probabilities.
        """
        # separate dataset with index
        if sp.issparse(X):
            left_feat = X[:, :self._split_index]
            right_feat = X[:, self._split_index:]
        else:
            left_feat, right_feat = np.hsplit(X, [self._split_index])

        return self._calc_probs(self.left_params[0], self.left_params[1], left_feat) * \
            self._calc_probs(self.right_params[0], self.right_params[1], right_feat)

    def predict(self, X: Matrix) -> np.array:
        """Return predicted labels.

        :Parameters:

        X: {array-like, sparse matrix} of shape (n_samples, n_left and right latent features)
            Feature matrix derived from concatenating left latent features with right latent features.

        :return:

        Predicted labels.
        """
        # calculating probs
        probs = self.predict_proba(X)
        return np.array([p >= 0.5 for p in probs])
