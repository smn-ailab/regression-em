"""test for RegressionEM."""
from typing import Union

from regression_em import RegressionEM
import numpy as np
import scipy.sparse as sp
from pytest import approx

# Define type.
Matrix = Union[np.ndarray, sp.csr_matrix]


class TestRegressionEM():
    """Check thata the model can work."""

    def setup(self):
        """Initialize hyper parameters.

        : Parameters:

        X: {np.array}
        Feature matrix for tests.
        X_sp: {csr matrix}
        y: {np.array}
        Labels for tests
        y_multi: {np.array}
        Labels to check that the model can handle multilabel.
        """
        self.rem = RegressionEM(split_index=2, max_iter=10, class_weights='balanced', alpha=1, epsilon=10**-10)
        self.rem_wo_class_weights = RegressionEM(split_index=2, max_iter=10, alpha=1, epsilon=10**-10)
        self.left_feat = np.array([[1, 0], [0, 1], [1, 1]])
        self.right_feat = np.array([[1, 0], [0, 1], [1, 1]])
        self.X = np.hstack([self.left_feat, self.right_feat])
        self.left_feat_sp = sp.csr_matrix(self.left_feat)
        self.right_feat_sp = sp.csr_matrix(self.right_feat)
        self.X_sp = sp.csr_matrix(self.X)
        self.y = np.array([0, 1, 1])
        self.y_multi = np.array([0, 1, 2])

    def test_integers_to_bools(self):
        """Test to convert in the following cases.

        1. 0/1 label
        2. multi class label
        """
        ans = [False, True, True]
        assert ans == self.rem._integers_to_bools(self.y)
        assert ans == self.rem._integers_to_bools(self.y_multi)

    def test_calc_probs(self):
        """Test to calculate probabilities in the following input cases.

        1. np.array
        2. csr_matrix
        """
        coef = np.array([1, 1])
        intercept = 0.5
        ans = np.array([0.817574, 0.817574, 0.924142])
        assert ans == approx(self.rem._calc_probs(coef, intercept, self.left_feat), rel=1e-4)
        assert ans == approx(self.rem._calc_probs(coef, intercept, self.left_feat_sp), rel=1e-4)

    def test_calc_logits(self):
        """Test to calculating logits in the following cases.

        1. prob =! 0 and prob =! 1
        2. prob = 0
        3. prob = 1
        """
        prob = np.array([0.5, 0, 1])
        ans = np.array([0, -23.0258509298, 23.0258509298])

        assert ans == approx(self.rem._calc_logits(prob), rel=1e-4)

    def test_calc_responsibility(self):
        """Test to calculate responsibilities in the following cases.

        1. target_param =! 1 or ref_param =! 1 and label = False
        2. target_param = 1 and ref_param = 1 and label = False -> 1 (avoid zero devision)
        3. label = True -> 1
        (see eq.1 of https: // static.googleusercontent.com/media/research.google.com/ja//pubs/archive/46485.pdf)
        """
        target_prob = [0.4, 1, 0.1]
        ref_prob = [0.5, 1, 0.9]
        is_positive = [False, False, True]
        ans = [0.25, 1, 1]
        for i in range(len(target_prob)):
            assert ans[i] == self.rem._calc_responsibility(target_prob[i], ref_prob[i], is_positive[i])

    def test_update_responsibilities(self):
        """Test to calculate.

        1. target_param =! 1 or ref_param =! 1 and label = False
        2. target_param = 1 and ref_param = 1 and label = False -> 1 (avoid zero devision)
        3. label = True -> 1
        """
        left_param = (np.array([1, 1]), 0.5)
        right_param = (np.array([1, 1]), 0.5)
        label = np.array([False, True, True])
        ans = np.array([0.4498160735, 1, 1])
        res = self.rem._update_responsibilities(left_param, self.left_feat, right_param, self.right_feat, label)
        res_sp = self.rem._update_responsibilities(left_param, self.left_feat_sp, right_param, self.right_feat_sp, label)
        assert ans[0] == approx(res[0], rel=1e-4)
        assert ans[0] == approx(res_sp[0], rel=1e-4)
        assert ans[1] == res[1]
        assert ans[1] == res_sp[1]
        assert ans[2] == res[2]
        assert ans[2] == res_sp[2]

    def test_update_params(self):
        """Test that the model can work with or without alpha."""
        responsibilities = np.array([0.1, 0.2, 0.3])
        sample_weights = np.array([3, 1.5, 1.5])

        rem_w_alpha = RegressionEM(split_index=2, max_iter=10, class_weights='balanced', alpha=1, epsilon=10**-10)
        assert rem_w_alpha._update_params(self.X, responsibilities, sample_weights)

        rem_wo_alpha = RegressionEM(split_index=2, max_iter=10, class_weights='balanced', epsilon=10**-10)
        assert rem_wo_alpha._update_params(self.X, responsibilities, sample_weights)

    def test_class_weights(self):
        """Test that the model can return accurate class_weights."""
        self.rem.fit(self.X, self.y)
        ans = [3, 1.5, 1.5]
        assert ans == approx(self.rem.sample_weights, rel=1e-4)

    def test_calc_log_likelihood(self):
        """Test that the model can work and handle 2 exceptions.

        exceptions
        1. The case that prob = 0 and label = False.
        2. The case that prob = 0 and label = True.
        """
        left_param = (np.array([1, 1]), 0.5)
        right_param = (np.array([1, 1]), 0.5)
        labels = np.array([True, True, False])
        ans = -2.73007
        assert ans == approx(self.rem._calc_log_likelihood(left_param, self.left_feat, right_param, self.right_feat, labels), rel=1e-4)

        # params for the case that prob = 0 and label = False.
        left_param_zero = (np.array([-10000, -10000]), -10000)
        right_param_zero = (np.array([-10000, -10000]), -10000)
        labels_zero = np.array([True, True, True])

        # params for the case that prob = 0 and label = True.
        left_param_one = (np.array([10000, 10000]), 10000)
        right_param_one = (np.array([10000, 10000]), 10000)
        labels_one = np.array([False, False, False])

        ans_exceptions = 3 * np.log(self.rem._epsilon)

        assert ans_exceptions == approx(self.rem._calc_log_likelihood(left_param_zero, self.left_feat, right_param_zero, self.right_feat, labels_zero), rel=1e-4)
        assert ans_exceptions == approx(self.rem._calc_log_likelihood(left_param_one, self.left_feat, right_param_one, self.right_feat, labels_one), rel=1e-4)

    def check_predictions(self, X: Matrix, y: np.array):
        """Check that the model is able to fit the classification data."""
        n_samples = len(y)
        labels = [False, True, True]
        classes = np.unique(labels)

        # with sample weights
        self.rem.fit(X, y)
        predicted = self.rem.predict(X)
        assert (np.unique(predicted) == classes).all()
        assert predicted.shape == (n_samples,)
        assert predicted == approx(y, rel=1e-4)

        # without sample weights
        self.rem_wo_class_weights.fit(X, y)
        predicted_wo_class_weights = self.rem_wo_class_weights.predict(X)
        assert (np.unique(predicted_wo_class_weights) == classes).all()
        assert predicted_wo_class_weights.shape == (n_samples,)
        assert predicted_wo_class_weights == approx(y, rel=1e-4)

    def test_fit_functions(self):
        """Test the fit functions."""
        TestRegressionEM.check_predictions(self, self.X, self.y)
        TestRegressionEM.check_predictions(self, self.X_sp, self.y)
