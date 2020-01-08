import scipy.sparse as sp
from sklearn.linear_model import LinearRegression


class RegressionEm:
    def fit(self):
        pass

    def predict(self):
        return 1


if __name__ == "__main__":
    rem = RegressionEm()

    x = sp.rand(100, 100)
    lr = LinearRegression()
    print(f"Prediction results: {rem.predict()}")
