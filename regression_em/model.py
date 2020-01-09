"""Package to perform regression EM algorithmself.
For the detail of this algortihm,
see [https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf].
"""
from random import random
from typing import Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

# Define type.
RegParam = Tuple[np.ndarray, float]


class RegressionEM:
    """Estimate latent factors based on below model.

       Outcome = Left latent * Right latent

       # TODO: Write detail description later.

       :Example:
       # TODO: Write usage later.
    """

    def __init__(self, alpha: float = 0, max_iter: int = 100, epsilon: float = 10 ** -10, with_sample_weights: bool = False) -> None:
        """Initialize hyper parameters."""
        self.alpha = alpha
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
        # targetのパラメーター更新のために更新後のrefでrefの確率を計算

        target_probs = self.calc_probs(target_params[0], target_params[1], target_feat)
        ref_probs = self.calc_probs(ref_params[0], ref_params[1], ref_feat)

        return np.vectorize(self.calc_responsibility)(target_probs, ref_probs, labels)

    def update_params(self, feat_mat: np.ndarray, responsibilities: np.ndarray, sample_weights) -> Tuple[np.ndarray, float]:
        """Return fitted Logistic Regression params.

        :param feat_mat: Feature matrix to be used to learn responsibility.
        :param responsibilities: Sequence of responsibilities calculated at E-step.
        :return: Updated M-step params.
        """
        if self.alpha:
            reg = Ridge(alpha=self.alpha)
        else:
            reg = LinearRegression()
        reg.fit(feat_mat, self.calc_logits(responsibilities), sample_weights)
        return reg.coef_, reg.intercept_

    def update_expo_params(self, feat_mat: np.ndarray, responsibilities: np.ndarray, sample_weights) -> Tuple[np.ndarray, float]:
        """Return fitted Logistic Regression params.

        :param feat_mat: Feature matrix to be used to learn responsibility.
        :param responsibilities: Sequence of responsibilities calculated at E-step.
        :return: Updated M-step params.
        """
        if self.alpha:
            reg = Ridge(alpha=self.alpha * 100)
        else:
            reg = LinearRegression()
        reg.fit(feat_mat, self.calc_logits(responsibilities), sample_weights)
        return reg.coef_, reg.intercept_

    def calc_log_likelihood(self, left_feat: np.ndarray, right_feat: np.ndarray,
                            labels: np.ndarray) -> np.ndarray:
        """Return log likelihood.

            # TODO: Describe formula here.

        :param left_feat:
        :param right_feat:
        :param labels:
        :return:
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

    def fit(self, left_feat: np.ndarray, right_feat: np.ndarray, labels: np.ndarray) -> None:
        """Estimate regression EM params.

        :param left_feat: Feature matrix to be used to learn left params.
        :param right_feat: Feature matrix to be used to learn right params.
        :param labels: Sequence of labels indicating each sample is positive or negative.
        """
        # Initialize params (feature weight, intercept).
        self.left_params = (np.random.rand(left_feat.shape[1]), random())
        self.right_params = (np.random.rand(right_feat.shape[1]), random())
        self.log_likelihoods = [self.calc_log_likelihood(left_feat, right_feat, labels)]
        max_ll = self.log_likelihoods[-1]
        best_left_params = self.left_params
        best_right_params = self.right_params

        sample_weights = None
        if self.with_sample_weights:
            pos_ratio = len([l for l in labels if l]) / len(labels)
            sample_weights = [1 / pos_ratio if l else 1 / (1 - pos_ratio) for l in labels]

        for epoch in range(self.max_iter):
            # Update left latent params
            left_responsibilities = self.update_responsibilities(self.left_params, left_feat, self.right_params, right_feat, labels)
            self.left_params = self.update_expo_params(left_feat, left_responsibilities, sample_weights)

            # Update right latent params
            right_responsibilities = self.update_responsibilities(self.right_params, right_feat, self.left_params, left_feat, labels)
            self.right_params = self.update_params(right_feat, right_responsibilities, sample_weights)

            # 尤度の計算と収束判定
            self.log_likelihoods.append(self.calc_log_likelihood(left_feat, right_feat, labels))

            if max_ll < self.log_likelihoods[-1]:
                max_ll = self.log_likelihoods[-1]
                best_left_params = self.left_params
                best_right_params = self.right_params

        self.left_params = best_left_params
        self.right_params = best_right_params

    def predict_proba(self, left_feat: np.ndarray, right_feat: np.ndarray) -> np.ndarray:
        """Return predicted probabilities.

        :param left_feat: Feature matrix to be used to predict left latent factor.
        :param right_feat: Feature matrix to be used to predict right latent factor.
        :return: Predicted probabilities.
        """
        return self.calc_probs(self.left_params[0], self.left_params[1], left_feat) * \
            self.calc_probs(self.right_params[0], self.right_params[1], right_feat)

    def predict(self, left_feat: np.ndarray, right_feat: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(left_feat, right_feat)
        return np.array([p >= 0.5 for p in probs])


if __name__ == "__main__":
    rem = RegressionEM(max_iter=20, with_sample_weights=True, alpha=1)
    print("test")
