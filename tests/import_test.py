"""Check if packages are appropriately installed."""

import numpy as np
from scipy.linalg.blas import ddot
from sklearn.linear_model import LinearRegression


def test_np_install():
    """Test numpy is installed."""
    assert 1 == np.ones(10)[0]


def test_blas_install():
    """Test blas is installed."""
    x = np.ones(3)
    y = np.ones(3)

    assert 3 == ddot(x, y)


def test_sklearn_install():
    """Test LinearRegression is installed."""
    X = np.random.rand(10, 10)
    y = np.random.rand(10, 1)

    lr = LinearRegression()
    lr.fit(X, y)
