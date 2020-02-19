"""test for RegressionEM."""
from typing import Union

from regression_em import RegressionEM
import numpy as np
import numpy.testing as npt
import scipy.sparse as sp

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
        assert ans == self.rem._integers_to_bools(self.y) and ans == self.rem._integers_to_bools(self.y_multi)

    def test_calc_probs(self):
        """Test to calculate probabilities in the following input cases.

        1. np.array
        2. csr_matrix
        """
        coef = np.array([1, 1])
        intercept = 0.5
        ans = np.array([0.817574, 0.817574, 0.924142])
        npt.assert_almost_equal(ans, self.rem._calc_probs(coef, intercept, self.left_feat), decimal=5)
        npt.assert_almost_equal(ans, self.rem._calc_probs(coef, intercept, self.left_feat_sp), decimal=5)

    def test_calc_logits(self):
        """Test to calculating logits in the following cases.

        1. prob =! 0 and prob =! 1
        2. prob = 0
        3. prob = 1
        """
        prob = np.array([0.5, 0, 1])
        ans = np.array([0, -23.0258509298, 23.0258509298])

        npt.assert_almost_equal(ans, self.rem._calc_logits(prob), decimal=7)

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
        npt.assert_almost_equal(ans[0], res[0], decimal=5)
        npt.assert_almost_equal(ans[0], res_sp[0], decimal=5)
        assert ans[1] == res[1] and ans[1] == res_sp[1]
        assert ans[2] == res[2] and ans[2] == res_sp[2]

    def test_update_params():
        """

        """

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

        # params for the case that prob = 0 and label = False.
        left_param_zero = (np.array([-10000, -10000]), -10000)
        right_param_zero = (np.array([-10000, -10000]), -10000)
        labels_zero = np.array([True, True, True])

        # params for the case that prob = 0 and label = True.
        left_param_one = (np.array([10000, 10000]), 10000)
        right_param_one = (np.array([10000, 10000]), 10000)
        labels_one = np.array([False, False, False])

        ans_exceptions = 3 * np.log(self.rem._epsilon)

        npt.assert_almost_equal(ans, self.rem._calc_log_likelihood(left_param, self.left_feat, right_param, self.right_feat, labels), decimal=4)
        npt.assert_almost_equal(ans_exceptions, self.rem._calc_log_likelihood(left_param_zero, self.left_feat, right_param_zero, self.right_feat, labels_zero), decimal=5)
        npt.assert_almost_equal(ans_exceptions, self.rem._calc_log_likelihood(left_param_one, self.left_feat, right_param_one, self.right_feat, labels_one), decimal=5)

    def check_predictions(self, X: Matrix, y: np.array):
        """Check that the model is able to fit the classification data."""
        n_samples = len(y)
        classes = np.unique(y)

        self.rem.fit(X, y)
        predicted = self.rem.predict(X)
        npt.assert_array_equal(np.unique(predicted), classes)

        assert predicted.shape == (n_samples,)
        npt.assert_array_equal(predicted, y)

    def test_fit_functions(self):
        """Test the fit functions."""
        TestRegressionEM.check_predictions(self, self.X, self.y)
        TestRegressionEM.check_predictions(self, self.X_sp, self.y)
